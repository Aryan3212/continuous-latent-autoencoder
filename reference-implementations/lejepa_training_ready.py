import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

"""
LEJEPA: Sketched Isotropic Gaussian Regularization (SIGReg)
===========================================================

Standalone implementation of the core LeJEPA loss, adapted to be faithful
to Algorithms 1 and 2 of the paper, and usable for speech/audio backbones
like Zipformer (sequence output (B, T, D)).

Components:
1. EppsPulley           -- univariate (1D) normality test.
2. SlicingUnivariateTest -- scales 1D tests to high-D via random slicing.
3. lejepa_loss          -- full paper-faithful Algorithm 2 (per-view SIGReg).
4. audio_lejepa_step    -- Zipformer-oriented wrapper that handles (B, T, D).
"""


def all_reduce_avg(x):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.AVG)
    return x


def world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class EppsPulley(nn.Module):
    """
    Epps-Pulley characteristic-function normality test (Algorithm 1).

    Faithfulness notes vs. paper:
    - Paper uses t = linspace(-5, 5, 17). We use the same symmetric domain
      by default (t_max=5) so the truncation matches the paper.
    - We split exp(i t x) into cos/sin for real-valued autograd stability;
      mathematically equivalent to the complex form.
    - err is weighted by the Gaussian window exp(-t^2/2), folded into the
      trapezoid weights for efficiency.
    - Final statistic scales with the *global* N = local_N * world_size,
      matching the paper's DDP-aware normalization.
    """

    def __init__(self, t_max: float = 5.0, n_points: int = 17):
        super().__init__()
        # Symmetric grid [-t_max, t_max] with n_points (paper default).
        t = torch.linspace(-t_max, t_max, n_points)
        dt = (2 * t_max) / (n_points - 1)

        # Trapezoidal integration weights: dt/2 at endpoints, dt interior.
        weights = torch.full((n_points,), dt)
        weights[0] = dt / 2
        weights[-1] = dt / 2

        # Target: characteristic function of N(0, 1).
        phi = torch.exp(-t.square() * 0.5)

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        # Pre-multiply integration weights by the Gaussian window (exp_f).
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., N, M) where N = batch samples, M = num slices.
        N_local = x.size(-2)
        N_global = N_local * world_size()

        # Broadcast samples onto integration grid: (..., N, M, T)
        x_t = x.unsqueeze(-1) * self.t

        # Empirical CF, averaged across batch (and DDP ranks).
        cos_mean = all_reduce_avg(torch.cos(x_t).mean(-3))
        sin_mean = all_reduce_avg(torch.sin(x_t).mean(-3))

        # |ecf - phi|^2 = (Re - phi)^2 + Im^2
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integration over t; scale by global N per the paper.
        return (err @ self.weights) * N_global


class SlicingUnivariateTest(nn.Module):
    """
    Slicing wrapper (Algorithm 1, sketching part).

    - A ~ N(0, I) projection matrix, columns L2-normalized.
    - Seeded by global_step so all DDP ranks share the same slices per step.
    - O(N * D * M) compute, O(N * M) memory.
    """

    def __init__(self, univariate_test: nn.Module, num_slices: int = 256):
        super().__init__()
        self.univariate_test = univariate_test
        self.num_slices = num_slices
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D)
        with torch.no_grad():
            seed = int(self.global_step.item())
            gen = torch.Generator(device=x.device).manual_seed(seed)
            A = torch.randn(
                x.size(-1), self.num_slices, device=x.device, generator=gen
            )
            A /= A.norm(p=2, dim=0)
            self.global_step += 1

        sliced = x @ A                        # (N, M)
        stats = self.univariate_test(sliced)  # (M,)
        return stats.mean()


# ==========================================
# PAPER-FAITHFUL LEJEPA LOSS (Algorithm 2)
# ==========================================

def lejepa_loss(
    g_emb: torch.Tensor,
    a_emb: torch.Tensor,
    bs: int,
    sigreg: SlicingUnivariateTest,
    lamb: float = 0.02,
):
    """
    Paper-faithful LeJEPA objective (Algorithm 2).

    Args:
        g_emb: embeddings of *global* views, shape (V_g * bs, K).
               Must be the concatenation of V_g view batches along dim 0.
               For non-ViT backbones (ResNet, Zipformer, ...), pass the same
               tensor as `a_emb` and set V_g == V_a (see paper note).
        a_emb: embeddings of *all* views, shape (V_a * bs, K).
        bs:    per-view batch size.
        sigreg: a SlicingUnivariateTest module (stateful, owns global_step).
        lamb:  trade-off between invariance and SIGReg. Paper default ~0.02.

    Returns:
        scalar loss.
    """
    K = g_emb.size(-1)

    # Per-sample prototype (center) across global views: (bs, K).
    centers = g_emb.view(-1, bs, K).mean(0)

    # All views reshaped: (V_a, bs, K).
    a_emb = a_emb.view(-1, bs, K)

    # Invariance: every view pulled to the shared center.
    sim = (centers - a_emb).square().mean()

    # SIGReg applied *per view* and averaged (paper: mean(SIGReg(emb) ...)).
    sigreg_vals = torch.stack([sigreg(emb) for emb in a_emb])
    sigreg_loss = sigreg_vals.mean()

    return (1 - lamb) * sim + lamb * sigreg_loss


# ==========================================
# AUDIO / ZIPFORMER INTEGRATION
# ==========================================

def pool_sequence(
    h: torch.Tensor,
    lengths: torch.Tensor | None = None,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Collapse Zipformer-style (B, T, D) output into (B, D) for SIGReg.

    Args:
        h:       (B, T, D) encoder output.
        lengths: optional (B,) valid-length tensor for masking padding.
        mode:    "mean" or "flatten". "flatten" returns (B*T_valid, D) for
                 frame-level regularization (stronger but frames are
                 temporally correlated).

    Returns:
        (B, D) if mode=="mean", else (B*T_valid, D).
    """
    B, T, D = h.shape
    if lengths is None:
        mask = torch.ones(B, T, device=h.device, dtype=h.dtype)
    else:
        idx = torch.arange(T, device=h.device).unsqueeze(0)
        mask = (idx < lengths.unsqueeze(1)).to(h.dtype)

    if mode == "mean":
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom

    if mode == "flatten":
        valid = mask.bool().reshape(-1)
        return h.reshape(B * T, D)[valid]

    raise ValueError(f"unknown pool mode: {mode}")


def audio_lejepa_step(
    encoder: nn.Module,
    views: list[torch.Tensor],
    lengths: list[torch.Tensor] | None,
    sigreg: SlicingUnivariateTest,
    lamb: float = 0.02,
    pool_mode: str = "mean",
):
    """
    Zipformer / wav2vec-style training step for LeJEPA.

    Audio-specific adaptations:
    - Zipformer is non-ViT, so per the paper we set global_views = all_views.
    - Encoder output is (B, T, D); we pool to (B, D) (or flatten to frames)
      before SIGReg. Mean-pool is the safer default; frame-level gives a
      larger effective N but frames are correlated.
    - "Views" are augmentation pairs of the same utterance:
      e.g., SpecAugment masks, additive noise, speed perturb, random crops.

    Args:
        encoder:  backbone producing (B, T, D) from raw features/waveform.
        views:    list of V view tensors, each shape (bs, ...), that the
                  encoder can consume. All views must have the same bs.
        lengths:  optional list of V length tensors, each shape (bs,), used
                  to mask padding before pooling. Pass None if unpadded.
        sigreg:   stateful SlicingUnivariateTest module.
        lamb:     SIGReg weight.
        pool_mode: "mean" (utterance-level) or "flatten" (frame-level).

    Returns:
        scalar loss.
    """
    V = len(views)
    bs = views[0].size(0)
    assert all(v.size(0) == bs for v in views), "all views need the same bs"

    # Single forward over the concatenated batch (matches Alg. 2's torch.cat).
    x_cat = torch.cat(views, dim=0)
    h_cat = encoder(x_cat)  # (V*bs, T, D)

    if lengths is not None:
        len_cat = torch.cat(lengths, dim=0)
    else:
        len_cat = None

    # Pool to (V*bs, D) for utterance-level, or (sum_frames, D) for frame.
    # SIGReg needs a per-view (N, D) split, so only "mean" supports the
    # full Algorithm 2 reshape cleanly. Frame mode regularizes the pooled
    # frame distribution globally instead.
    if pool_mode == "mean":
        emb = pool_sequence(h_cat, len_cat, mode="mean")  # (V*bs, D)
        return lejepa_loss(emb, emb, bs=bs, sigreg=sigreg, lamb=lamb)

    # Frame-level variant: invariance on utterance means, SIGReg on frames.
    utt = pool_sequence(h_cat, len_cat, mode="mean")      # (V*bs, D)
    K = utt.size(-1)
    centers = utt.view(V, bs, K).mean(0)
    sim = (centers - utt.view(V, bs, K)).square().mean()

    # Per-view frame-level SIGReg.
    sigreg_vals = []
    for v in range(V):
        h_v = h_cat[v * bs : (v + 1) * bs]                # (bs, T, D)
        len_v = len_cat[v * bs : (v + 1) * bs] if len_cat is not None else None
        frames = pool_sequence(h_v, len_v, mode="flatten")
        sigreg_vals.append(sigreg(frames))
    sigreg_loss = torch.stack(sigreg_vals).mean()

    return (1 - lamb) * sim + lamb * sigreg_loss


# ==========================================
# SANITY CHECK
# ==========================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) EppsPulley on samples from N(0, I) should give a low value.
    x_normal = torch.randn(512, 128)
    loss_fn = SlicingUnivariateTest(EppsPulley(), num_slices=256)
    print(f"SIGReg on N(0,I):      {loss_fn(x_normal).item():.4f}")

    # 2) On a non-Gaussian distribution, SIGReg should be much larger.
    x_uniform = (torch.rand(512, 128) - 0.5) * 3.46  # ~unit variance
    loss_fn2 = SlicingUnivariateTest(EppsPulley(), num_slices=256)
    print(f"SIGReg on Uniform:     {loss_fn2(x_uniform).item():.4f}")

    # 3) Fake Zipformer step: encoder returns (B, T, D).
    class FakeZipformer(nn.Module):
        def __init__(self, d=128):
            super().__init__()
            self.proj = nn.Linear(d, d)

        def forward(self, x):  # x: (B, T, d)
            return self.proj(x)

    enc = FakeZipformer(128)
    views = [torch.randn(8, 50, 128) for _ in range(2)]
    lengths = [torch.full((8,), 50, dtype=torch.long) for _ in range(2)]
    sigreg = SlicingUnivariateTest(EppsPulley(), num_slices=256)
    loss = audio_lejepa_step(enc, views, lengths, sigreg, lamb=0.02)
    print(f"audio_lejepa_step:     {loss.item():.4f}")

