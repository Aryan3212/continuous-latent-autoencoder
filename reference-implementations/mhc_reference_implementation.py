import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

"""
MANIFOLD-CONSTRAINED HYPER-CONNECTIONS (mHC) - REFERENCE IMPLEMENTATION
========================================================================

Theoretical Background:
-----------------------
mHC was introduced by DeepSeek (https://arxiv.org/abs/2512.24880) as an improvement 
over Hyper-Connections (https://arxiv.org/abs/2409.19606).

Key Concept:
Traditional Residual Connections (ResNet) use x_{l+1} = x_l + F(x_l).
Hyper-Connections generalize this to multiple residual streams (s streams).
mHC constrains the mixing matrices to lie on specific manifolds to preserve 
the "identity" nature and stability of residual flow in very deep models.

The mHC Update Rule:
    x_{l+1} = H_res * x_l + H_post * F(H_pre * x_l)

Constraints:
1. H_res: Doubly Stochastic Matrix (Birkhoff Polytope). 
   All entries >= 0, rows sum to 1, and columns sum to 1.
   This prevents signal explosion or decay across streams.
2. H_pre / H_post: Non-negative mixing maps (usually Softmax).

This file contains two implementations:
1. CleanMHC: Follows the paper math strictly for clarity.
2. FeatureRichMHC: Includes advanced options like Orthostochastic projection (Newton-Schulz).
"""

# ----------------------------------------------------------------------------
# Helper Functions: Projections to the Manifold
# ----------------------------------------------------------------------------

def sinkhorn_log(logits, num_iters=10, tau=0.05):
    """
    Projects a matrix onto the Birkhoff Polytope (doubly stochastic matrices)
    using the Sinkhorn-Knopp algorithm in log-space for numerical stability.
    """
    n = logits.shape[-1]
    Z = logits / tau
    # Log-marginal target is -log(n) for each row/column to sum to 1
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        # Alternative row and column normalization
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))

def newton_schulz_ortho(X, steps=5, eps=1e-7, coeffs=(3.0, -3.2, 1.2)):
    """
    Newton-Schulz iteration to compute the nearest orthogonal matrix.
    Used for orthostochastic projection.
    """
    a, b, c = coeffs
    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

def orthostochastic_project(logits, steps=5):
    """
    An orthostochastic matrix is the element-wise square of an orthogonal matrix.
    Every orthostochastic matrix is doubly stochastic.
    """
    O = newton_schulz_ortho(logits, steps=steps)
    return O.square()

# ----------------------------------------------------------------------------
# Implementation 1: CleanMHC (The "Math-First" Version)
# ----------------------------------------------------------------------------

class CleanMHC(nn.Module):
    """
    A clean, minimal implementation of mHC. 
    Strictly follows the core algorithm from the DeepSeek paper.
    """
    def __init__(self, num_streams, dim, branch=None):
        super().__init__()
        self.num_streams = num_streams
        self.branch = branch

        # H_res: Doubly stochastic mixing of residual streams
        # Initialized to identity (0 on diagonal, -8 elsewhere) for stable start
        init_h_res = torch.full((num_streams, num_streams), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)

        # H_pre: Softmax mixing of streams into the branch input
        # Initialized to pick one stream (the first one) initially
        init_h_pre = torch.full((num_streams,), -8.0)
        init_h_pre[0] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)

        # H_post: Softmax weights to distribute branch output back to streams
        self.H_post_logits = nn.Parameter(torch.zeros(num_streams))

    def forward(self, x):
        # 1. Project logits to the doubly stochastic manifold
        H_res = sinkhorn_log(self.H_res_logits)
        
        # 2. Mix residual streams for the NEXT state
        # x shape: (batch, streams, seq, dim)
        x_res = einsum(H_res, x, 's t, b s n d -> b t n d')

        # 3. Prepare input for the branch F
        # H_pre is normalized via softmax to ensure convex combination
        h_pre = F.softmax(self.H_pre_logits, dim=-1)
        branch_in = einsum(h_pre, x, 's, b s n d -> b n d')

        # 4. Compute branch output
        # Usually an Attention or MLP layer
        branch_out = self.branch(branch_in) if self.branch else branch_in

        # 5. Distribute branch output back to streams via H_post
        h_post = F.softmax(self.H_post_logits, dim=-1)
        # branch_out: (batch, n, d) -> distributed: (batch, streams, n, d)
        branch_dist = einsum(h_post, branch_out, 's, b n d -> b s n d')

        # 6. Final mHC update
        return x_res + branch_dist

# ----------------------------------------------------------------------------
# Implementation 2: FeatureRichMHC (The "Research-Ready" Version)
# ----------------------------------------------------------------------------

class FeatureRichMHC(nn.Module):
    """
    The version used in the repository's benchmarks.
    Includes stability tricks and alternative projections.
    """
    def __init__(
        self, 
        num_streams, 
        dim, 
        branch=None,
        proj_type="sinkhorn", # or "orthostochastic"
        identity_mix=True,     # H_res = (1-alpha)I + alpha*S
        alpha_init=0.01        # Strength of mixing at start
    ):
        super().__init__()
        self.num_streams = num_streams
        self.branch = branch
        self.proj_type = proj_type
        
        # Parameters same as CleanMHC
        self.H_res_logits = nn.Parameter(torch.full((num_streams, num_streams), -8.0).fill_diagonal_(0.0))
        self.H_pre_logits = nn.Parameter(torch.full((num_streams,), -8.0).fill_diagonal_(0.0))
        self.H_post_logits = nn.Parameter(torch.zeros(num_streams))

        # Stability Trick: Identity Mix
        # Ensures that even if Sinkhorn is noisy, there is a hard identity path
        self.identity_mix = identity_mix
        if identity_mix:
            # Learned alpha via sigmoid to keep it in (0, 1)
            logit_alpha = math.log(alpha_init / (1 - alpha_init))
            self.H_res_alpha_logit = nn.Parameter(torch.tensor(logit_alpha))

    def forward(self, x):
        # Select Projection Method
        if self.proj_type == "orthostochastic":
            S = orthostochastic_project(self.H_res_logits)
        else:
            S = sinkhorn_log(self.H_res_logits)

        # Apply Identity Mix
        if self.identity_mix:
            alpha = torch.sigmoid(self.H_res_alpha_logit)
            I = torch.eye(self.num_streams, device=x.device)
            H_res = (1 - alpha) * I + alpha * S
        else:
            H_res = S

        # Residual Mixing
        x_res = einsum(H_res, x, 's t, b s n d -> b t n d')

        # Branch Path
        h_pre = F.softmax(self.H_pre_logits, dim=-1)
        branch_in = einsum(h_pre, x, 's, b s n d -> b n d')
        
        branch_out = self.branch(branch_in) if self.branch else branch_in
        
        h_post = F.softmax(self.H_post_logits, dim=-1)
        branch_dist = einsum(h_post, branch_out, 's, b n d -> b s n d')

        return x_res + branch_dist

# ----------------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Settings
    B, S, N, D = 2, 4, 128, 64 # Batch, Streams, SeqLen, Dim
    dummy_input = torch.randn(B, S, N, D)
    
    # Minimal Branch (e.g., a simple linear layer)
    simple_branch = nn.Linear(D, D)

    # Initialize MHC
    mhc_layer = FeatureRichMHC(num_streams=S, dim=D, branch=simple_branch)
    
    output = mhc_layer(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if H_res is doubly stochastic
    with torch.no_grad():
        S_mat = sinkhorn_log(mhc_layer.H_res_logits)
        print(f"\nRow sums (should be 1): {S_mat.sum(dim=-1)}")
        print(f"Col sums (should be 1): {S_mat.sum(dim=-2)}")
