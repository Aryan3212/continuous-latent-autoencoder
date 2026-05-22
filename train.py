from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Any, Dict, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import torch.nn.functional as F

from data.augment import FeatureMaskConfig, MixConfig, WaveAugConfig, apply_feature_mask, apply_waveform_augment, maybe_mix_pair
from data.dataset import WebDatasetConfig, get_audio_wds, collate_fixed
from eval.inline_probe import InlineProbe
from losses.multires_stft import MultiResSTFTConfig, MultiResSTFTLoss
from models.decoder_generator import DecoderConfig, WaveformDecoder
from models.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)
from models.encoder import Encoder, EncoderConfig
from models.frontend_conv import ConvFrontend, FrontendConfig
from models.mhc import sinkhorn_log
from models.projector import Projector, ProjectorConfig
from models.sigreg import SIGReg, SIGRegConfig
from optim.lr_schedulers import Eden, Eden2
from optim.scaled_adam import ScaledAdam
from utils.checkpoint import save_checkpoint, save_run_metadata, try_git_hash
from utils.config import apply_overrides, load_config
from utils.logging import JsonlLogger, maybe_init_wandb
from utils.seed import seed_all


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _select_device(cfg: Dict[str, Any]) -> torch.device:
    want = (cfg.get("run") or {}).get("device", "auto")
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _latent_noise_sigma(cfg: Dict[str, Any], step: int, device: torch.device) -> torch.Tensor:
    ncfg = cfg.get("latent_noise") or {}
    if not ncfg.get("enabled", False):
        return torch.tensor(0.0, device=device)
    warmup = int(ncfg.get("warmup_steps", 0))
    if step < warmup:
        return torch.tensor(0.0, device=device)
    kind = str(ncfg.get("kind", "uniform"))
    if kind == "fixed":
        return torch.tensor(float(ncfg.get("noise_tau", ncfg.get("sigma_max", 0.05))), device=device)
    sigma_max = float(ncfg.get("sigma_max", 0.05))
    return torch.empty((), device=device).uniform_(0.0, sigma_max)


def _pool_utt(z: torch.Tensor) -> torch.Tensor:
    # z: (B, D, T) -> (B, D) utterance-level mean pool.
    return z.mean(dim=-1)


def _lejepa_invariance(z_cat: torch.Tensor, num_views: int, return_per_sample: bool = False) -> torch.Tensor:
    """
    Algorithm 2 invariance term: pull every view to the shared center
    where center = mean of (utterance-pooled) view embeddings across views.
    No stop-gradient; anti-collapse is provided by averaging.
    z_cat: (V*B, D, T) concatenated view embeddings.
    """
    V = num_views
    VB, D, T = z_cat.shape
    B = VB // V
    emb = _pool_utt(z_cat).view(V, B, D)          # (V, B, D)
    centers = emb.mean(dim=0)                     # (B, D)
    per_view = (centers.unsqueeze(0) - emb).square().mean(dim=-1)  # (V, B)
    if return_per_sample:
        return per_view.mean(dim=0)               # (B,)
    return per_view.mean()


def _dense_jepa_loss(
    z_cat: torch.Tensor,
    mask_cat: torch.Tensor,
    num_views: int,
    lam_context: float = 1.0,
    distance_weight: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    V-JEPA 2.1 'Dense Predictive Loss' adapted to audio (1-D time).

    For every frame t and every view v, pull z[v,b,:,t] toward the per-frame
    center across views: center[b,:,t] = mean_v z[v,b,:,t]. Apply two
    weight schedules:
      L_predict  : on frames the SpecAug time-mask zeroed out (weight = 1).
      L_context  : on the surviving (visible) frames, weighted by
                   lam_context / sqrt(d_min(t)) where d_min is the distance
                   in frames to the nearest masked position in the same view.
                   This is V-JEPA 2.1's masks_dist.py rule, 1-D specialisation.

    z_cat:    (V*B, D, T)   concatenated view embeddings.
    mask_cat: (V*B, 1, T)   1 at masked positions, 0 elsewhere
                            (output of apply_feature_mask).

    Returns (loss, l_predict_scalar, l_context_scalar) for logging.
    """
    V = num_views
    VB, D, T = z_cat.shape
    B = VB // V
    z = z_cat.view(V, B, D, T)
    centers = z.mean(dim=0, keepdim=True)              # (1, B, D, T)
    err = (z - centers).pow(2).mean(dim=2)             # (V, B, T)  per-frame MSE

    m = mask_cat.view(V, B, T).to(z.dtype)             # 1 on masked frames
    one_minus_m = 1.0 - m

    # L_predict: average over masked frames.
    pred_num = (err * m).sum()
    pred_den = m.sum().clamp_min(1.0)
    l_predict = pred_num / pred_den

    # Distance-weighted L_context: for each (v,b,t) with m=0, find min |t - t'|
    # over t' where m[v,b,t']=1. Done via two passes of a parallel scan.
    if distance_weight:
        big = float(T + 1)
        # Forward pass: distance to nearest masked frame at index <= t.
        d_fwd = torch.full_like(err, big)
        d_bwd = torch.full_like(err, big)
        # Vectorised: walk through T, accumulate.
        # d_fwd[..., t] = 0 if m[t] else d_fwd[..., t-1] + 1
        prev = torch.full((V, B), big, device=err.device, dtype=err.dtype)
        for t in range(T):
            prev = torch.where(m[..., t] > 0.5, torch.zeros_like(prev), prev + 1.0)
            d_fwd[..., t] = prev
        nxt = torch.full((V, B), big, device=err.device, dtype=err.dtype)
        for t in range(T - 1, -1, -1):
            nxt = torch.where(m[..., t] > 0.5, torch.zeros_like(nxt), nxt + 1.0)
            d_bwd[..., t] = nxt
        d_min = torch.minimum(d_fwd, d_bwd).clamp_min(1.0)  # avoid div-by-zero
        ctx_w = one_minus_m / d_min.sqrt()
    else:
        ctx_w = one_minus_m

    ctx_num = (err * ctx_w).sum()
    ctx_den = ctx_w.sum().clamp_min(1e-6)
    l_context = ctx_num / ctx_den

    loss = l_predict + lam_context * l_context
    return loss, l_predict.detach(), l_context.detach()


def _pool(z: torch.Tensor) -> torch.Tensor:
    # z: (B,d,T') -> (B,3d)
    avg_p = z.mean(dim=-1)
    std_p = z.std(dim=-1, unbiased=False)
    max_p = z.max(dim=-1)[0]
    return torch.cat([avg_p, std_p, max_p], dim=1)


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _encode(model: torch.nn.ModuleDict, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    h0 = model["frontend"](wav)
    hE = model["encoder"](h0)
    return h0, hE, hE, {}


def _decode(model: torch.nn.ModuleDict, z: torch.Tensor, target_len: int, sigma: torch.Tensor) -> torch.Tensor:
    z_dec = z + torch.randn_like(z) * sigma
    return model["decoder"](z_dec, target_len=target_len)


def _primary_infonce(z_a: torch.Tensor, z_mask: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    # In-batch InfoNCE over pooled utterance embeddings.
    # Positives: clean vs masked view of the same utterance. Negatives: other utterances in batch.
    e_all = F.normalize(_pool(z_a), dim=-1)          # (B, 3D)
    e_pos = F.normalize(_pool(z_mask), dim=-1)       # (B, 3D)
    B = e_all.size(0)
    sim = e_all @ e_all.t() / temp                   # (B, B)
    eye_mask = torch.eye(B, dtype=torch.bool, device=e_all.device)
    pos = (e_all * e_pos).sum(dim=-1, keepdim=True) / temp  # (B, 1)
    neg_inf = torch.finfo(sim.dtype).min
    logits = torch.cat([pos, sim.masked_fill(eye_mask, neg_inf)], dim=1)  # (B, 1+B)
    target = torch.zeros(B, dtype=torch.long, device=e_all.device)
    return F.cross_entropy(logits, target)


def main() -> None:
    # Hardware acceleration flags
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # Limit PyTorch to 95% of physical VRAM to prevent WSL2/Windows shared memory slowdowns
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--log_interval_steps", type=int, default=None)
    ap.add_argument("--eval_interval_steps", type=int, default=None)
    ap.add_argument("--save_interval_steps", type=int, default=None)
    ap.add_argument("--run_eval_on_save", action="store_true")
    ap.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler with W&B")
    ap.add_argument("--profile_wait", type=int, default=0, help="Steps to wait before profiling")
    ap.add_argument("--profile_warmup", type=int, default=0, help="Steps to warm up profiler")
    ap.add_argument("--profile_active", type=int, default=1, help="Steps to actively profile")
    ap.add_argument("overrides", nargs="*")
    args, unknown = ap.parse_known_args()

    # Convert CLI flags like --optim.lr 0.001 or --optim.lr=0.001 to overrides optim.lr=0.001
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            # Handle --key=value
            if "=" in arg:
                key, val = arg[2:].split("=", 1)
                args.overrides.append(f"{key}={val}")
                i += 1
                continue
            
            # Handle --key value
            key = arg[2:]
            if i + 1 < len(unknown):
                val = unknown[i + 1]
                if not val.startswith("-"):
                    args.overrides.append(f"{key}={val}")
                    i += 2
                    continue
            
            # If we get here, it's a flag without a value or boolean flag not supported
            print(f"Warning: dangling flag {arg} ignored or boolean not supported")
            i += 1
        else:
            # Positional arg in unknown?
            print(f"Warning: unknown argument {arg}")
            i += 1

    cfg = apply_overrides(load_config(args.config), args.overrides)
    cfg["_resolved_config_path"] = args.config

    # Resume path layout: <out_dir>/<run_id>/checkpoints/<name>.pt
    # Infer the existing run_id so we reuse its out_dir and wandb run.
    if args.resume:
        resume_path = pathlib.Path(args.resume)
        if resume_path.parent.name == "checkpoints":
            inferred_run_id = resume_path.parent.parent.name
            if not cfg["run"].get("run_id"):
                cfg["run"]["run_id"] = inferred_run_id

    seed_all(int(cfg["run"]["seed"]))
    device = _select_device(cfg)

    run_id = cfg["run"].get("run_id") or _now_run_id()
    out_root = pathlib.Path(cfg["run"]["out_dir"]) / run_id
    ckpt_dir = out_root / "checkpoints"
    log_dir = out_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jsonl = JsonlLogger(str(log_dir / "train.jsonl"))
    wb = maybe_init_wandb(cfg, run_id, str(out_root), resume=bool(args.resume))
    
    # Initialize CodeCarbon
    codecarbon_tracker = None
    if cfg.get("run", {}).get("track_emissions", True):
        try:
            from codecarbon import EmissionsTracker
            codecarbon_tracker = EmissionsTracker(
                output_dir=str(out_root),
                output_file="emissions.csv",
                log_level="error" # reduce spam
            )
            codecarbon_tracker.start()
        except ImportError:
            pass

    # Data
    dcfg = cfg["data"]
    if dcfg.get("train_manifest") is None:
        raise ValueError("Set data.train_manifest to the shard URL pattern (e.g. data/shards/train/train-{0000..0150}.tar)")
    meta_extra = {
        "git_hash": try_git_hash(cwd=str(pathlib.Path(".").resolve())),
        "train_manifest": str(dcfg["train_manifest"]),
        "val_manifest": str(dcfg.get("val_manifest") or ""),
    }
    save_run_metadata(str(out_root), cfg, extra=meta_extra)
    train_ds = get_audio_wds(
        WebDatasetConfig(
            urls=dcfg["train_manifest"],
            sample_rate=int(dcfg["sample_rate"]),
            segment_seconds=float(dcfg["segment_seconds"]),
            shuffle_size=int(dcfg.get("shuffle_size", 1000)),
        )
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=bool(dcfg.get("pin_memory", True)),
        persistent_workers=bool(dcfg.get("persistent_workers", False)) if int(dcfg.get("num_workers", 4)) > 0 else False,
        collate_fn=collate_fixed,
        drop_last=True,
    )
    # Note: WebDataset natively handles Distributed Data Parallel sharding via worker and node splitting.
    # We do NOT use DistributedSampler as IterableDatasets do not support length or explicit indices.

    mix_cfg = MixConfig(**(cfg.get("aug", {}).get("mix", {}) or {}))

    # Model
    mcfg = cfg["model"]
    frontend = ConvFrontend(FrontendConfig(**mcfg["frontend"]))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg["encoder"]))
    
    latent_dim = int(mcfg["encoder"]["d_model"])
    
    decoder_cfg = DecoderConfig(**mcfg["decoder"])
    decoder = WaveformDecoder(latent_dim, decoder_cfg)
    if decoder_cfg.latent_stats_path:
        stats = torch.load(decoder_cfg.latent_stats_path, map_location="cpu")
        decoder.set_latent_stats(stats["mean"], stats["var"])
    proj_cfg_raw = (mcfg.get("projector") or {}).copy()
    projector_cfg = ProjectorConfig(**{k: v for k, v in proj_cfg_raw.items() if k in {"hidden_dim", "output_dim", "n_hidden_layers"}})
    projector = Projector(latent_dim, projector_cfg)
    proj_dim = projector.output_dim

    sigreg_cfg = cfg["loss"]["sigreg"].copy()
    _allowed = {"num_slices", "t_max", "n_points"}
    sigreg_cfg = {k: v for k, v in sigreg_cfg.items() if k in _allowed}
    sigreg = SIGReg(proj_dim, SIGRegConfig(**sigreg_cfg))

    model = torch.nn.ModuleDict(
        {
            "frontend": frontend,
            "encoder": encoder,
            "projector": projector,
            "decoder": decoder,
            "sigreg": sigreg,
        }
    ).to(device)

    gan_cfg = cfg.get("gan") or {}
    gan_enabled = bool(gan_cfg.get("enabled", False))
    discriminators = None
    d_optimizer = None
    if gan_enabled:
        mpd = MultiPeriodDiscriminator(
            periods=gan_cfg.get("periods", [2, 3, 5, 7, 11]),
            channels=int(gan_cfg.get("mpd_channels", 32)),
        )
        msd = MultiScaleDiscriminator(
            scales=int(gan_cfg.get("msd_scales", 3)),
            channels=int(gan_cfg.get("msd_channels", 16)),
        )
        discriminators = torch.nn.ModuleDict({"mpd": mpd, "msd": msd}).to(device)
        
        d_optimizer = torch.optim.AdamW(
            discriminators.parameters(),
            lr=float(gan_cfg.get("d_lr", 2.0e-4)),
            betas=tuple(gan_cfg.get("d_betas", [0.8, 0.99])),
            weight_decay=float(gan_cfg.get("d_weight_decay", 0.0)),
        )

    # Losses
    stft = MultiResSTFTLoss(MultiResSTFTConfig(**cfg["loss"]["stft"])).to(device)
    feat_mask_cfg = FeatureMaskConfig(**(cfg.get("aug", {}).get("feature_mask", {}) or {}))
    wave_aug_cfg = WaveAugConfig(**(cfg.get("aug", {}).get("wave_aug", {}) or {}))

    # Optim
    ocfg = cfg["optim"]
    if ocfg["kind"] == "scaled_adam":
        optimizer = ScaledAdam(
            model.parameters(),
            lr=float(ocfg["lr"]),
            clipping_scale=ocfg.get("clipping_scale"),
            betas=tuple(ocfg["betas"]),
            eps=float(ocfg["eps"]),
            scalar_lr_scale=float(ocfg.get("scalar_lr_scale", 0.1)),
            param_min_rms=float(ocfg.get("param_min_rms", 1.0e-5)),
            param_max_rms=float(ocfg.get("param_max_rms", 3.0)),
            scalar_max=float(ocfg.get("scalar_max", 10.0)),
            size_update_period=int(ocfg.get("size_update_period", 4)),
            clipping_update_period=int(ocfg.get("clipping_update_period", 100)),
        )
    elif ocfg["kind"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(ocfg["lr"]),
            betas=tuple(ocfg["betas"]),
            eps=float(ocfg["eps"]),
            weight_decay=float(ocfg["weight_decay"]),
        )
    else:
        raise ValueError(f"Unknown optimizer kind: {ocfg['kind']}")

    scheduler = None
    scfg = ocfg.get("scheduler") or {}
    if scfg.get("kind") == "eden":
        scheduler = Eden(
            optimizer,
            lr_batches=float(scfg.get("lr_batches", 5000)),
            lr_epochs=float(scfg.get("lr_epochs", 6)),
            warmup_batches=float(scfg.get("warmup_batches", 500)),
            warmup_start=float(scfg.get("warmup_start", 0.5)),
            verbose=bool(scfg.get("verbose", False)),
        )
    elif scfg.get("kind") == "eden2":
        scheduler = Eden2(
            optimizer,
            lr_batches=float(scfg.get("lr_batches", 5000)),
            warmup_batches=float(scfg.get("warmup_batches", 500)),
            warmup_start=float(scfg.get("warmup_start", 0.5)),
            verbose=bool(scfg.get("verbose", False)),
        )
    elif scfg.get("kind") == "cosine":
        warmup_steps = int(scfg.get("warmup_steps", 2000))
        total_steps = int(scfg.get("total_steps", 200000))
        min_lr_ratio = float(scfg.get("min_lr_ratio", 0.0))
        base_lr = float(ocfg["lr"])
        _cosine_inner = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=base_lr * min_lr_ratio,
        )
        _warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(1, warmup_steps),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[_warmup, _cosine_inner],
            milestones=[warmup_steps],
        )
        scheduler.step_batch = lambda step=None: scheduler.step()
        scheduler.step_epoch = lambda epoch=None: None
        scheduler.state_dict = scheduler.state_dict
        scheduler.load_state_dict = scheduler.load_state_dict

    use_amp = bool(cfg["run"].get("amp", True)) and device.type == "cuda"
    g_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    d_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    resume_best: Dict[str, float] | None = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if gan_enabled and (state.get("extra") or {}).get("discriminators"):
            discriminators.load_state_dict(state["extra"]["discriminators"], strict=True)
            if d_optimizer and (state.get("extra") or {}).get("d_optimizer"):
                d_optimizer.load_state_dict(state["extra"]["d_optimizer"])
        if state.get("scaler") and g_scaler.is_enabled():
            g_scaler.load_state_dict(state["scaler"])
        if state.get("d_scaler") and d_scaler.is_enabled():
            d_scaler.load_state_dict(state["d_scaler"])
        if scheduler and (state.get("extra") or {}).get("scheduler"):
            scheduler.load_state_dict(state["extra"]["scheduler"])
        step = int(state.get("step", 0))
        resume_best = (state.get("extra") or {}).get("best")

    # CLI overrides for loop intervals.
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
    if args.log_interval_steps is not None:
        cfg["train"]["log_interval_steps"] = int(args.log_interval_steps)
    if args.eval_interval_steps is not None:
        cfg["train"]["eval_interval_steps"] = int(args.eval_interval_steps)
    if args.save_interval_steps is not None:
        cfg["train"]["save_interval_steps"] = int(args.save_interval_steps)
    if args.run_eval_on_save:
        cfg["train"]["run_eval_on_save"] = True

    # Training loop
    model.train()
    max_steps = int(cfg["train"]["max_steps"])
    grad_accum = int(cfg["train"]["grad_accum_steps"])
    grad_clip = float(cfg["optim"]["grad_clip"])
    val_batches = cfg["train"].get("val_batches")
    if val_batches is not None:
        val_batches = int(val_batches)

    jcfg = cfg["loss"]["jepa"]
    jepa_w = float(jcfg["weight"])
    num_views = int(jcfg.get("num_views", 2))
    if num_views < 2:
        raise ValueError(f"loss.jepa.num_views must be >= 2 (Algorithm 2 needs averaging); got {num_views}")
    sig_w = float(cfg["loss"]["sigreg"]["weight"])
    stft_w = float(cfg["loss"].get("stft_weight", 1.0))
    wav_l1_w = float(cfg["loss"].get("wav_l1_weight", 0.0))

    mix_recon_cfg = cfg["loss"].get("mix_recon") or {}
    mix_recon_enabled = bool(mix_recon_cfg.get("enabled", False))
    mix_recon_w = float(mix_recon_cfg.get("weight", 1.0))
    mix_recon_start = int(mix_recon_cfg.get("start_step", 0))

    primary_cfg = cfg["loss"].get("primary") or {}
    primary_enabled = bool(primary_cfg.get("enabled", False))
    primary_w = float(primary_cfg.get("weight", 0.0))
    primary_temp = float(primary_cfg.get("temp", 0.1))

    mix_view_w = float(jcfg.get("mix_view_weight", 1.0))
    # V-JEPA 2.1 context-loss weight: relative weight of L_context vs L_predict
    # in the Dense Predictive Loss. Paper uses ~1.0 with distance weighting.
    lam_context_w = float(jcfg.get("context_weight", 1.0))
    gan_start = int(gan_cfg.get("start_step", 0))
    gan_warmup_steps = int(gan_cfg.get("warmup_steps", 5000))

    best: Dict[str, float] = {"val_jepa": float("inf"), "asr_wer": float("inf"), "composite": -float("inf")}
    if isinstance(resume_best, dict):
        for k in ["val_jepa", "asr_wer", "composite"]:
            if k in resume_best:
                best[k] = float(resume_best[k])

    inline_probe = InlineProbe(
        cfg_probe=(cfg.get("loss") or {}).get("inline_probe") or {},
        sample_rate=int(dcfg["sample_rate"]),
        encoder_dim=latent_dim,
        device=device,
        out_root=out_root,
    )

    def _validate_one() -> Dict[str, float]:
        if not dcfg.get("val_manifest"):
            return {}
        val_ds = get_audio_wds(
            WebDatasetConfig(
                urls=dcfg["val_manifest"],
                sample_rate=int(dcfg["sample_rate"]),
                segment_seconds=float(dcfg["segment_seconds"]),
                random_crop=False, # use start crop for validation consistency
                resampled=False,
                shuffle_size=0,
            )
        )
        val_ds = val_ds.batched(int(cfg["train"]["batch_size"]), collation_fn=collate_fixed)
        val_dl = torch.utils.data.DataLoader(
            val_ds,
            batch_size=None,
            num_workers=0,
        )
        model.eval()
        sums = {
            "val_stft": torch.tensor(0.0, device=device),
            "val_jepa": torch.tensor(0.0, device=device),
            "val_sig": torch.tensor(0.0, device=device),
        }
        n = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for vb in val_dl:
                vw = vb["wav"].to(device)
                h0, hE, z, stats = _encode(model, vw)
                h0m, mask = apply_feature_mask(h0, feat_mask_cfg)
                hEm = model["encoder"](h0m)
                zm = hEm
                # Validation uses two views (clean and masked) for monitoring.
                # JEPA + SIGReg act on projected output to mirror training.
                p_clean = model["projector"](z)
                p_mask = model["projector"](zm)
                p_pair = torch.cat([p_clean, p_mask], dim=0)
                v_jepa = _lejepa_invariance(p_pair, num_views=2)
                def _flatten(t: torch.Tensor) -> torch.Tensor:
                    return t.permute(0, 2, 1).reshape(-1, t.size(1))
                fp = _flatten(p_clean); fpm = _flatten(p_mask)
                v_sig_a_f, _ = sigreg(fp, step=step); v_sig_a_f = v_sig_a_f / max(1, fp.size(0))
                v_sig_m_f, _ = sigreg(fpm, step=step); v_sig_m_f = v_sig_m_f / max(1, fpm.size(0))
                v_sig = 0.5 * (v_sig_a_f + v_sig_m_f)
                xh = _decode(model, z, target_len=vw.size(-1), sigma=torch.tensor(0.0, device=device))
                v_stft, _ = stft(xh, vw)
                sums["val_stft"] += v_stft.detach()
                sums["val_jepa"] += v_jepa.detach()
                sums["val_sig"] += v_sig.detach()
                n += 1
                if val_batches is not None and n >= val_batches:
                    break
        model.train()
        return {k: (v / max(1, n)).item() for k, v in sums.items()}

    def _extra_state(**extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"best": dict(best), "scheduler": scheduler.state_dict() if scheduler else None}
        if gan_enabled and discriminators is not None:
            payload["discriminators"] = discriminators.state_dict()
            payload["d_optimizer"] = d_optimizer.state_dict() if d_optimizer else None
        payload["d_scaler"] = d_scaler.state_dict() if d_scaler.is_enabled() else None
        payload.update(extra)
        return payload

    prof = None
    if args.profile:
        import wandb
        if wandb.run is None:
            print("Warning: W&B is not initialized, logging profiler trace to local './profiler_logs' directory.")
            handler = torch.profiler.tensorboard_trace_handler('./profiler_logs')
        else:
            handler = wandb.profiler.torch_trace_handler()

        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=args.profile_wait, warmup=args.profile_warmup, active=args.profile_active, repeat=1),
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

    accum_stats: Dict[str, torch.Tensor] = {}

    epochs = max_steps  # Safe upper bound since we break at max_steps
    for epoch in range(epochs):
        if scheduler is not None and epoch > 0:
            scheduler.step_epoch(epoch)
        
        train_it = iter(train_dl)
        
        epoch_done = False
        while not epoch_done and step < max_steps:
            optimizer.zero_grad(set_to_none=True)
            if gan_enabled and d_optimizer is not None:
                d_optimizer.zero_grad(set_to_none=True)
            
            total_loss = torch.tensor(0.0, device=device)
            mb_stats: Dict[str, Any] = {}

            microbatches = []
            for micro in range(grad_accum):
                try:
                    batch = next(train_it)
                except StopIteration:
                    epoch_done = True
                    break
                
                try:
                    batch_b = next(train_it)
                except StopIteration:
                    train_it = iter(train_dl)
                    batch_b = next(train_it)
                
                microbatches.append((batch, batch_b))

            if not microbatches:
                break

            for i_mb, (batch, batch_b) in enumerate(microbatches):
                wav_a = batch["wav"]  # (B,1,T)
                dataset_names = [m.get("dataset", "unknown") for m in batch["meta"]]
                wav_b = batch_b["wav"]

                # Build mix waveform + per-sample primary target when enabled.
                mixed_mask = torch.zeros((wav_a.size(0),), dtype=torch.bool, device=device)
                primary_idx = torch.zeros((wav_a.size(0),), dtype=torch.long, device=device)
                snr_db_vals = torch.zeros((wav_a.size(0),), dtype=torch.float32, device=device)
                wav_mix = wav_a
                wav_tgt = wav_a
                has_mix = False

                if mix_cfg.enabled and mix_cfg.prob > 0.0:
                    wav_mix_list = []
                    wav_tgt_list = []
                    for i in range(wav_a.size(0)):
                        y, did, sdb, pidx = maybe_mix_pair(wav_a[i, 0], wav_b[i, 0], mix_cfg)
                        if did:
                            has_mix = True
                        mixed_mask[i] = did
                        primary_idx[i] = pidx
                        snr_db_vals[i] = float(sdb)
                        wav_mix_list.append(y)
                        wav_tgt_list.append(wav_a[i, 0] if pidx == 0 else wav_b[i, 0])
                    wav_mix = torch.stack(wav_mix_list, dim=0).unsqueeze(1).to(device, non_blocking=True)
                    wav_tgt = torch.stack(wav_tgt_list, dim=0).unsqueeze(1).to(device, non_blocking=True)
                else:
                    wav_mix = wav_mix.to(device, non_blocking=True)
                    wav_tgt = wav_tgt.to(device, non_blocking=True)

                wav_a = wav_a.to(device, non_blocking=True)
                wav_b = wav_b.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    # --- Algorithm 2 (LeJEPA): V independently-augmented views,
                    #     single fused forward, invariance to their averaged center.
                    B = wav_a.size(0)
                    sr = int(dcfg["sample_rate"])
                    view_wavs = [apply_waveform_augment(wav_a, sr, wave_aug_cfg) for _ in range(num_views)]
                    wav_cat = torch.cat(view_wavs, dim=0)  # (V*B, 1, T)

                    h0_cat = model["frontend"](wav_cat)
                    # Feature-mask applied per-sample on the fused batch (independent masks per view).
                    h0_masked, time_mask = apply_feature_mask(h0_cat, feat_mask_cfg)
                    z_cat = model["encoder"](h0_masked)  # (V*B, D, T')

                    # ---- Projector (LeJEPA / LeWM) ----------------------------------
                    # Decouple loss-space from representation-space: JEPA + SIGReg act
                    # on p_cat; decoder + probes keep reading z_cat. BatchNorm in the
                    # projector is what lets SIGReg actually reshape the distribution
                    # to N(0, I); applying it directly on the BiasNorm-conditioned
                    # encoder output resists Gaussianisation.
                    p_cat = model["projector"](z_cat)  # (V*B, P, T')

                    # ---- Invariance objective ---------------------------------------
                    # V-JEPA 2.1 'Dense Predictive Loss' adapted to audio:
                    #   per-frame MSE between each view and the cross-view center,
                    #   split into L_predict (masked frames) + λ * L_context (visible
                    #   frames, weighted 1/sqrt(d_min) to enforce local coherence).
                    l_jepa_mask, l_jepa_pred_dbg, l_jepa_ctx_dbg = _dense_jepa_loss(
                        p_cat,
                        time_mask if time_mask.shape[-1] == p_cat.shape[-1] else
                            F.interpolate(time_mask, size=p_cat.shape[-1], mode="nearest"),
                        num_views=num_views,
                        lam_context=lam_context_w,
                        distance_weight=True,
                    )

                    # ---- SIGReg (LeWM-style, frame-level only, on projector out) ---
                    # Reshape p_cat (V*B, P, T') -> (T'*V*B, P) so SIGReg treats each
                    # (frame, view, sample) triple as an independent point. Paper
                    # N-scaling is left to SIGReg itself; no extra /N rescaling here.
                    D_lat = p_cat.size(1)
                    sig_input = p_cat.permute(2, 0, 1)                # (T', V*B, P)
                    l_sig, sig_stats_last = sigreg(
                        sig_input.reshape(-1, D_lat),                 # (T'*V*B, P)
                        step=step,
                    )
                    sig_stats = {
                        "sigreg_view": l_sig.detach(),
                        "l_sig_utt": torch.tensor(0.0, device=device),
                        "l_sig_frm": l_sig.detach(),
                        "l_jepa_predict": l_jepa_pred_dbg,
                        "l_jepa_context": l_jepa_ctx_dbg,
                        "z_var_min": sig_stats_last["z_var_min"],
                        "z_var_med": sig_stats_last["z_var_med"],
                        "z_var_max": sig_stats_last["z_var_max"],
                    }

                    z_a = z_cat[:B]               # view-0 encoder embeddings (decoder + rank diag)
                    z_mask = z_cat[B : 2 * B]     # view-1 encoder embeddings (rank diag only)
                    p_a = p_cat[:B]               # view-0 projected (JEPA-space diagnostics)
                    p_mask = p_cat[B : 2 * B]     # view-1 projected (JEPA-space diagnostics)

                    l_jepa_mix = torch.tensor(0.0, device=device)
                    l_primary = torch.tensor(0.0, device=device)
                    l_stft_mix = torch.tensor(0.0, device=device)

                    # Decode from view-0 to clean wav_a (denoising reconstruction).
                    sigma = _latent_noise_sigma(cfg, step, device)
                    x_hat = _decode(model, z_a, target_len=wav_a.size(-1), sigma=sigma)

                    l_stft_ps, stft_stats_ps = stft(x_hat, wav_a, return_per_sample=True)
                    l_stft = l_stft_ps.mean()
                    stft_stats = {k: v.mean().detach() for k, v in stft_stats_ps.items()}

                    l_wav_ps = (x_hat - wav_a).abs().mean(dim=(1, 2))
                    l_wav = l_wav_ps.mean()

                    l_g_adv = torch.tensor(0.0, device=device)
                    l_fm = torch.tensor(0.0, device=device)
                    l_d = torch.tensor(0.0, device=device)
                    gan_w = 0.0

                # GAN: forward passes run in AMP (float16 activations to save VRAM),
                # but losses are cast to float32 to avoid hinge-loss overflow.
                if gan_enabled and discriminators is not None and step >= gan_start:
                    gan_progress = min(1.0, (step - gan_start) / max(1, gan_warmup_steps))

                    # --- Discriminator step (AMP forward, float32 loss) ---
                    _set_requires_grad(discriminators, True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        d_real_mpd, fmap_real_mpd = discriminators["mpd"](wav_a)
                        d_fake_mpd, _ = discriminators["mpd"](x_hat.detach())
                        d_real_msd, fmap_real_msd = discriminators["msd"](wav_a)
                        d_fake_msd, _ = discriminators["msd"](x_hat.detach())
                    # Loss in float32 (discriminator_loss casts logits via .float())
                    l_d = 0.5 * (
                        discriminator_loss(d_real_mpd, d_fake_mpd)
                        + discriminator_loss(d_real_msd, d_fake_msd)
                    )
                    d_scaler.scale(l_d / grad_accum).backward()
                    _set_requires_grad(discriminators, False)

                    # --- Generator adversarial step (AMP forward, float32 loss) ---
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        d_fake_mpd_g, fmap_fake_mpd_g = discriminators["mpd"](x_hat)
                        d_fake_msd_g, fmap_fake_msd_g = discriminators["msd"](x_hat)
                    fmap_real_mpd_det = [[f.detach() for f in layer] for layer in fmap_real_mpd]
                    fmap_real_msd_det = [[f.detach() for f in layer] for layer in fmap_real_msd]
                    l_g_adv = 0.5 * (
                        generator_loss(d_fake_mpd_g) + generator_loss(d_fake_msd_g)
                    )
                    l_fm = 0.5 * (
                        feature_matching_loss(fmap_real_mpd_det, fmap_fake_mpd_g)
                        + feature_matching_loss(fmap_real_msd_det, fmap_fake_msd_g)
                    )

                    # Adaptive GAN weight (VQGAN-style) in float32
                    last_layer = model["decoder"].out_conv.weight
                    rec_grads = torch.autograd.grad(
                        stft_w * l_stft.float(), last_layer, retain_graph=True
                    )[0].float()
                    gan_grads = torch.autograd.grad(
                        l_g_adv + l_fm, last_layer, retain_graph=True
                    )[0].float()
                    d_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + 1e-4)
                    d_weight = torch.clamp(d_weight, 0.0, 10.0).detach()
                    gan_w = d_weight.item() * gan_progress

                with torch.amp.autocast("cuda", enabled=use_amp):
                    l_jepa = l_jepa_mask + mix_view_w * l_jepa_mix

                    loss = (
                        stft_w * l_stft
                        + wav_l1_w * l_wav
                        + jepa_w * l_jepa
                        + sig_w * l_sig
                        + gan_w * (l_g_adv + l_fm)
                        + (mix_recon_w * l_stft_mix if (mix_recon_enabled and step >= mix_recon_start) else 0.0)
                        + (primary_w * l_primary if primary_enabled else 0.0)
                    )
                    loss = loss / grad_accum

                g_scaler.scale(loss).backward()
                total_loss = total_loss + loss.detach()

                # Diagnostic metrics for collapse detection
                with torch.no_grad():
                    # Flattened full rank (existing metric, keep for compatibility)
                    z_flat = z_a.permute(0, 2, 1).reshape(-1, z_a.size(1))   # (B*T, D)
                    z_centered = z_flat - z_flat.mean(dim=0)
                    z_cov = (z_centered.T @ z_centered) / (z_flat.size(0) - 1)
                    z_eigvals = torch.linalg.eigvalsh(z_cov)
                    z_rank = (z_eigvals.sum()**2) / (z_eigvals.pow(2).sum() + 1e-8)

                    # Utterance-level rank: pool over time, measure between-utterance diversity
                    z_utt = z_a.mean(dim=2)                                   # (B, D)
                    z_utt_c = z_utt - z_utt.mean(dim=0)
                    z_utt_cov = (z_utt_c.T @ z_utt_c) / max(z_utt.size(0) - 1, 1)
                    z_utt_eig = torch.linalg.eigvalsh(z_utt_cov)
                    z_rank_utt = (z_utt_eig.sum()**2) / (z_utt_eig.pow(2).sum() + 1e-8)

                    # Residual rank: frame-level variation after removing utterance mean
                    z_res = z_a - z_a.mean(dim=2, keepdim=True)               # (B, D, T)
                    z_res_flat = z_res.permute(0, 2, 1).reshape(-1, z_a.size(1))
                    z_res_cov = (z_res_flat.T @ z_res_flat) / max(z_res_flat.size(0) - 1, 1)
                    z_res_eig = torch.linalg.eigvalsh(z_res_cov)
                    z_rank_res = (z_res_eig.sum()**2) / (z_res_eig.pow(2).sum() + 1e-8)

                    # JEPA collapse detector: if both views collapse to zero,
                    # l_jepa_mask looks small but actually signals failure.
                    # jepa_to_norm_ratio > 0.1 means views genuinely differ.
                    # Measured in PROJECTOR space (where JEPA acts), not encoder space.
                    z_a_rms = z_a.pow(2).mean().sqrt()
                    z_mask_rms = z_mask.pow(2).mean().sqrt()
                    p_a_rms = p_a.pow(2).mean().sqrt()
                    p_mask_rms = p_mask.pow(2).mean().sqrt()
                    jepa_diff_rms = (p_a - p_mask).pow(2).mean().sqrt()
                    jepa_to_norm_ratio = jepa_diff_rms / p_a_rms.clamp_min(1e-6)

                mb_step_stats = {
                    "l_stft": l_stft.detach(),
                    "l_wav": l_wav.detach(),
                    "l_jepa": l_jepa.detach(),
                    "l_jepa_mask": l_jepa_mask.detach(),
                    "l_sig": l_sig.detach(),
                    "z_rank": z_rank.detach(),
                    "z_rank_utt": z_rank_utt.detach(),
                    "z_rank_res": z_rank_res.detach(),
                    "z_a_rms": z_a_rms.detach(),
                    "z_mask_rms": z_mask_rms.detach(),
                    "p_a_rms": p_a_rms.detach(),
                    "p_mask_rms": p_mask_rms.detach(),
                    "jepa_diff_rms": jepa_diff_rms.detach(),
                    "jepa_to_norm_ratio": jepa_to_norm_ratio.detach(),
                    "sigma": sigma.detach(),
                    "z_mean": z_a.mean().detach(),
                    "z_std": z_a.std(unbiased=False).detach(),
                    "vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
                }
                mb_step_stats.update({k: v.detach() for k, v in stft_stats.items()})
                mb_step_stats.update({k: v.detach() for k, v in sig_stats.items()})

                if mix_cfg.enabled:
                    mb_step_stats.update({
                        "l_jepa_mix": l_jepa_mix.detach(),
                        "l_stft_mix": l_stft_mix.detach(),
                        "mixed_frac": mixed_mask.float().mean().detach(),
                        "snr_db_mean": (snr_db_vals * mixed_mask.float()).sum().detach() / mixed_mask.float().sum().detach().clamp(min=1e-8),
                    })
                    if primary_enabled:
                        mb_step_stats["l_primary"] = l_primary.detach()

                if gan_enabled:
                    mb_step_stats.update({
                        "l_g_adv": l_g_adv.detach(),
                        "l_fm": l_fm.detach(),
                        "l_d": l_d.detach(),
                        "gan_w": float(gan_w),
                    })

                unique_ds = set(dataset_names)
                if len(unique_ds) > 1:
                    for ds_name in unique_ds:
                        indices = [i for i, n in enumerate(dataset_names) if n == ds_name]
                        if not indices:
                            continue
                        idx_t = torch.tensor(indices, device=device)
                        mb_step_stats[f"loss_stft/{ds_name}"] = l_stft_ps[idx_t].mean().detach()
                        mb_step_stats[f"loss_wav/{ds_name}"] = l_wav_ps[idx_t].mean().detach()

                for k, v in mb_step_stats.items():
                    if k not in mb_stats:
                        mb_stats[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
                    else:
                        mb_stats[k] = mb_stats[k] + (v if isinstance(v, torch.Tensor) else torch.tensor(v, device=device))

            n_mb = len(microbatches)
            stats = {k: v / n_mb for k, v in mb_stats.items()}
            stats["loss"] = total_loss

            # Optimize
            if grad_clip and grad_clip > 0:
                g_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            g_scaler.step(optimizer)
            g_scaler.update()
            if gan_enabled and d_optimizer is not None and step >= gan_start:
                if grad_clip and grad_clip > 0:
                    d_scaler.unscale_(d_optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminators.parameters(), grad_clip)
                d_scaler.step(d_optimizer)
                d_scaler.update()


            for k, v in stats.items():
                if k not in accum_stats:
                    accum_stats[k] = torch.tensor(0.0, device=device)
                if isinstance(v, torch.Tensor):
                    accum_stats[k] += v.detach()
                else:
                    accum_stats[k] += torch.tensor(v, device=device)

            step += 1
            if scheduler is not None:
                scheduler.step_batch(step)

            # Inline detached CTC probe — diagnoses latent quality without
            # waiting for the offline ASR eval (eval_interval_steps). Encoder
            # forward runs under torch.no_grad inside InlineProbe; only the
            # probe's linear head is trained.
            inline_probe.step(model, step, use_amp)
            inline_probe.maybe_emit(step, wb)

            log_interval = int(cfg["train"]["log_interval_steps"])
            if step % log_interval == 0:
                log_stats = {}
                for k, v in accum_stats.items():
                    log_stats[k] = v.item() / log_interval
                    v.zero_()
                epoch_idx = scheduler.epoch if scheduler is not None else epoch
                row = {"step": step, "epoch": epoch_idx, **log_stats}

                encoder_mod = model.get("encoder") if isinstance(model, dict) else None
                if encoder_mod is not None and hasattr(encoder_mod, "mhc_wrappers"):
                    for wrapper in encoder_mod.mhc_wrappers:
                        if not hasattr(wrapper, "H_res_alpha_logit"):
                            continue
                        l_idx = getattr(wrapper, "layer_index", -1)
                        with torch.no_grad():
                            alpha = torch.sigmoid(wrapper.H_res_alpha_logit).item()
                            row[f"mhc/layer_{l_idx}_alpha"] = alpha
                            S = sinkhorn_log(
                                wrapper.H_res_logits,
                                num_iters=wrapper.mhc_num_iters,
                                tau=wrapper.mhc_tau,
                            )
                            # Row entropy of doubly-stochastic mixing matrix:
                            # ~0 = identity (no mixing), log(num_streams) = uniform mixing.
                            row_ent = -(S * (S.clamp_min(1e-12)).log()).sum(dim=-1).mean().item()
                            row[f"mhc/layer_{l_idx}_S_row_entropy"] = row_ent

                jsonl.log(row)
                if wb is not None:
                    wb.log(row, step=step)

            if step % int(cfg["train"]["save_interval_steps"]) == 0:
                last_path = str(ckpt_dir / "last.pt")
                save_checkpoint(
                    last_path,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scaler=g_scaler if g_scaler.is_enabled() else None,
                    cfg=cfg,
                    extra=_extra_state(),
                )

                # Optionally run probes on the just-saved checkpoint.
                if bool(cfg.get("eval", {}).get("enabled", False)) and bool(cfg["train"].get("run_eval_on_save", False)):
                    from eval.run_probes import run_all_probes

                    print(f"[{time.strftime('%H:%M:%S')}] Starting evaluation block at step {step}...", flush=True)
                    eval_start_t = time.perf_counter()
                    
                    # Move everything to CPU to free GPU memory for subprocesses
                    model.cpu()
                    def optimizer_to(optim, device):
                        for state in optim.state.values():
                            for k, v in state.items():
                                if torch.is_tensor(v):
                                    state[k] = v.to(device)
                    optimizer_to(optimizer, "cpu")
                    if discriminators is not None:
                        discriminators.cpu()
                    if d_optimizer is not None:
                        optimizer_to(d_optimizer, "cpu")
                    
                    # Stop profiler to flush its memory and stop recording
                    if prof is not None:
                        print(f"[{time.strftime('%H:%M:%S')}] Pausing profiler for evaluation...", flush=True)
                        prof.stop()
                    
                    torch.cuda.empty_cache()

                    try:
                        results = run_all_probes(
                            run_dir=str(out_root),
                            step=step,
                            exp_cfg=cfg,
                            ckpt_path=last_path,
                            python_bin=sys.executable,
                        )
                    finally:
                        # Restore to GPU regardless of eval success/failure
                        print(f"[{time.strftime('%H:%M:%S')}] Restoring model to GPU...", flush=True)
                        model.to(device)
                        optimizer_to(optimizer, device)
                        if discriminators is not None:
                            discriminators.to(device)
                        if d_optimizer is not None:
                            optimizer_to(d_optimizer, device)
                        
                        torch.cuda.empty_cache()
                        if prof is not None:
                            print(f"[{time.strftime('%H:%M:%S')}] Resuming profiler...", flush=True)
                            prof.start()

                    eval_elapsed = time.perf_counter() - eval_start_t
                    print(f"[{time.strftime('%H:%M:%S')}] Evaluation block finished in {eval_elapsed:.2f}s", flush=True)

                    row = {"step": step, "probe": results}
                    jsonl.log(row)
                    if wb is not None:
                        to_log: Dict[str, Any] = {}
                        asr = results.get("asr") or {}
                        emo = results.get("emotion") or {}
                        gen = results.get("gender") or {}

                        if asr.get("train", {}).get("wer") is not None:
                            to_log["probe/asr_wer_train"] = float(asr["train"]["wer"])
                        if asr.get("dev", {}).get("wer") is not None:
                            to_log["probe/asr_wer_dev"] = float(asr["dev"]["wer"])
                            
                        if asr.get("dev", {}).get("examples"):
                            import wandb
                            cols = ["Ref", "Hyp"]
                            data = [[ex["ref"], ex["hyp"]] for ex in asr["dev"]["examples"]]
                            to_log["probe/asr_examples"] = wandb.Table(columns=cols, data=data)

                        if emo.get("accuracy") is not None:
                            to_log["probe/emotion_accuracy"] = float(emo["accuracy"])
                        if emo.get("macro_f1") is not None:
                            to_log["probe/emotion_macro_f1"] = float(emo["macro_f1"])

                        if gen.get("accuracy") is not None:
                            to_log["probe/gender_accuracy"] = float(gen["accuracy"])
                        
                        if "visualization" in results:
                            import wandb
                            to_log["probe/latents"] = wandb.Image(results["visualization"], caption=f"Step {step} Latents")

                        # Log probe timing so we can track eval duration on W&B
                        to_log["probe/total_time_s"] = eval_elapsed
                        probe_timing = results.get("_timing") or {}
                        for probe_name, t_sec in probe_timing.items():
                            safe_name = probe_name.lower().replace(" ", "_")
                            to_log[f"probe/time_{safe_name}_s"] = abs(t_sec)
                            if t_sec < 0:
                                to_log[f"probe/failed_{safe_name}"] = 1

                        if to_log:
                            wb.log(to_log, step=step)

                    # best_asr / best_composite
                    asr_wer = results.get("asr", {}).get("dev", {}).get("wer")
                    if asr_wer is not None and float(asr_wer) < best["asr_wer"]:
                        best["asr_wer"] = float(asr_wer)
                        save_checkpoint(
                            str(ckpt_dir / "best_asr.pt"),
                            step=step,
                            model=model,
                            optimizer=optimizer,
                            scaler=g_scaler if g_scaler.is_enabled() else None,
                            cfg=cfg,
                            extra=_extra_state(probe=results),
                        )

                    composite = 0.0
                    if "asr" in results and "dev" in results["asr"]:
                        composite += -float(results["asr"]["dev"]["wer"])
                    if "emotion" in results:
                        composite += float(results["emotion"].get("macro_f1", 0.0))
                    if "gender" in results:
                        composite += float(results["gender"].get("accuracy", 0.0))
                    if composite > best["composite"]:
                        best["composite"] = composite
                        save_checkpoint(
                            str(ckpt_dir / "best_composite.pt"),
                            step=step,
                            model=model,
                            optimizer=optimizer,
                            scaler=g_scaler if g_scaler.is_enabled() else None,
                            cfg=cfg,
                            extra=_extra_state(probe=results, composite=composite),
                        )

            if step % int(cfg["train"]["eval_interval_steps"]) == 0:
                v = _validate_one()
                if v:
                    row = {"step": step, **v}
                    jsonl.log(row)
                    if wb is not None:
                        wb.log(row, step=step)

                    if v.get("val_jepa", float("inf")) < best["val_jepa"]:
                        best["val_jepa"] = float(v["val_jepa"])
                        save_checkpoint(
                            str(ckpt_dir / "best_jepa.pt"),
                            step=step,
                            model=model,
                            optimizer=optimizer,
                            scaler=g_scaler if g_scaler.is_enabled() else None,
                            cfg=cfg,
                            extra=_extra_state(val=v),
                        )

                # Probes are triggered on save (run_eval_on_save), not on eval.

            if prof is not None:
                prof.step()

    if prof is not None:
        prof.stop()

    save_checkpoint(
        str(ckpt_dir / "last.pt"),
        step=step,
        model=model,
        optimizer=optimizer,
        scaler=g_scaler if g_scaler.is_enabled() else None,
        cfg=cfg,
        extra=_extra_state(),
    )
    
    if codecarbon_tracker is not None:
        emissions: float = codecarbon_tracker.stop()
        if wb is not None:
            wb.log({"emissions_kg_co2": emissions}, step=step)


if __name__ == "__main__":
    main()
