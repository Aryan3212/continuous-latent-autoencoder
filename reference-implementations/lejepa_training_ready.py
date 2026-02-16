import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

"""
LEJEPA: Sketched Isotropic Gaussian Regularization (SIGReg)
===========================================================

This file contains a standalone implementation of the core LeJEPA loss.
The goal of this loss is to force learned embeddings to follow a 
Standard Isotropic Gaussian distribution N(0, I) without using 
heuristics like stop-gradients or teacher-student networks.

There are two main components here:
1. EppsPulley: A univariate (1D) normality test.
2. SlicingUnivariateTest: A wrapper that scales 1D tests to high dimensions.
"""

def all_reduce_avg(x):
    """Utility for Distributed Data Parallel (DDP) training."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.AVG)
    return x

class EppsPulley(nn.Module):
    """
    The 'Heart' of SIGReg: The Epps-Pulley Test.
    
    WHY USE THIS?
    Classical tests (like Jarque-Bera) use moments like x^3 and x^4, which are 
    unstable and can cause 'exploding gradients' in Deep Learning.
    
    Epps-Pulley uses the Characteristic Function: cos(tx) and sin(tx). 
    Because sines and cosines are bounded between [-1, 1], the gradients are 
    extremely stable, making it perfect for training in bfloat16/float16.
    """
    def __init__(self, t_max=3, n_points=17):
        super().__init__()
        # We integrate the difference between the empirical and theoretical 
        # characteristic functions. n_points=17 is the paper's default.
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        
        # Integration weights (Trapezoidal rule)
        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt 
        
        # Precompute the 'target' (the characteristic function of a Normal distribution)
        phi = torch.exp(-t.square() * 0.5)
        
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x):
        # x shape: [..., NumSamples, NumSlices]
        N = x.size(-2)
        
        # Project samples onto our integration points t
        x_t = x.unsqueeze(-1) * self.t
        
        # Compute Empirical Characteristic Function (Real and Imaginary parts)
        cos_mean = all_reduce_avg(torch.cos(x_t).mean(-3))
        sin_mean = all_reduce_avg(torch.sin(x_t).mean(-3))

        # Squared distance from the target Normal distribution
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integration across the 't' points
        return (err @ self.weights) * N

class SlicingUnivariateTest(nn.Module):
    """
    The 'Strategy' of SIGReg: Random Slicing.
    
    WHY USE THIS?
    Testing for multivariate normality in 1024 dimensions is mathematically 
    expensive (O(N^2)). 
    
    Instead, we use 'Slicing':
    1. Project the 1024-D data onto many random 1D lines (slices).
    2. If every 1D projection is Gaussian, the whole high-D cloud is Gaussian.
    
    This turns an O(N^2) problem into an O(N) problem, making it scale 
    linearly with batch size and dimensions.
    """
    def __init__(self, univariate_test, num_slices=1024):
        super().__init__()
        self.univariate_test = univariate_test
        self.num_slices = num_slices
        # Global step helps keep random projections synchronized across GPUs
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        # x shape: [Batch, Dim]
        with torch.no_grad():
            # Ensure all GPUs use the same random slices for this step
            seed = self.global_step.item()
            gen = torch.Generator(device=x.device).manual_seed(seed)
            
            # Create random projection matrix A
            # Every column is a random unit vector (a 'slice')
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=gen)
            A /= A.norm(p=2, dim=0)
            self.global_step += 1

        # Project high-D data to 1D slices: [Batch, Dim] @ [Dim, Slices] -> [Batch, Slices]
        sliced_data = x @ A
        
        # Apply the 1D test to all slices and return the mean
        stats = self.univariate_test(sliced_data)
        return stats.mean()

# ==========================================
# EXAMPLE USAGE IN A TRAINING LOOP
# ==========================================

def example_training_step(model, x_views, lamb=0.02):
    """
    The 'Lean' way: Multi-view Invariance (as seen in MINIMAL.md)
    
    LeJEPA is 'Lean' because it doesn't require a separate predictor 
    or teacher-student network. It just uses a single encoder.
    
    Args:
        model: Your backbone + projector
        x_views: A tensor of shape [NumViews, Batch, C, H, W]
                 (e.g., 2 or 8 different crops of the same images)
        lamb: The trade-off between Invariance and Regularization
    """
    # 1. Forward pass for all views
    # Output shape: [NumViews, Batch, Dim]
    embeddings = model(x_views)
    
    # 2. Invariance Loss
    # We want all views of the same image to map to the same point.
    # We compare each view to the average of all views.
    mean_embedding = embeddings.mean(dim=0)
    inv_loss = (mean_embedding - embeddings).square().mean()
    
    # 3. SIGReg Loss (The code implemented above)
    # This prevents 'collapse' (model outputting zeros for everything)
    # By forcing the target representations to be Isotropic Gaussian.
    sigreg_fn = SlicingUnivariateTest(EppsPulley(), num_slices=1024)
    
    # We regularize the embeddings (usually the mean or the first view)
    sigreg_loss = sigreg_fn(mean_embedding)
    
    # 4. Final Objective
    total_loss = (1 - lamb) * inv_loss + lamb * sigreg_loss
    
    return total_loss

if __name__ == "__main__":
    # Quick sanity check
    test_input = torch.randn(32, 128) # Batch 32, Dim 128
    loss_fn = SlicingUnivariateTest(EppsPulley())
    
    loss = loss_fn(test_input)
    print(f"Computed SIGReg Loss: {loss.item():.4f}")
    print("If input is perfectly Normal, loss should be low.")
