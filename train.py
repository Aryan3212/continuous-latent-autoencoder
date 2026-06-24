from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist

from data_loading import (
    apply_waveform_augment,
    apply_waveform_chunk_mask,
    make_frame_chunk_masks,
)
from data_loading import AudioDataset, DatasetConfig, collate_fixed
from losses import (
    MultiResSTFTLoss,
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
)
from models.decoder_generator import WaveformDecoder
from models.discriminator import MultiPeriodDiscriminator
from models.encoder import Encoder
from models.frontend_conv import ConvFrontend
from models.mhc import sinkhorn_log
from models.projector import Projector
from models.sigreg import SIGReg
from config import apply_overrides, load_config
from schema import Config


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module if DDP-wrapped, else the module itself."""
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        return module.module
    return module


def _clean_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    """state_dict with any DDP ``module.`` prefix stripped, so checkpoints written
    under DDP are byte-compatible with single-GPU ones (and resume either way)."""
    return {k.replace(".module.", ".", 1): v for k, v in model.state_dict().items()}


def _gather_with_grad(x: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """All-gather ``x`` (N_local, D) across ranks into (N_global, D).

    Other ranks' rows are gathered as constants; the local rank's slot keeps its
    autograd graph, so backward delivers gradients only to the local samples. The
    caller compensates DDP's 1/world_size gradient averaging by scaling the loss
    term by world_size, recovering the single-GPU full-batch gradient. No-op when
    world_size == 1.
    """
    if world_size == 1:
        return x
    xc = x.contiguous()
    gathered = [torch.empty_like(xc) for _ in range(world_size)]
    dist.all_gather(gathered, xc)
    gathered[rank] = x  # restore autograd on the local slot
    return torch.cat(gathered, dim=0)


class JsonlLogger:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def maybe_init_wandb(cfg: Config, run_id: str, run_dir: str, resume: bool = False):
    wb = cfg.run.wandb
    if not wb.enabled:
        return None
    import wandb
    return wandb.init(
        project=wb.project,
        name=wb.name or run_id,
        id=run_id,
        resume="allow" if resume else None,
        dir=run_dir,
        config=cfg.model_dump(),
    )


def save_checkpoint(
    path: str,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Config,
    extra: Optional[Dict[str, Any]] = None,
    disc: Optional[torch.nn.Module] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
    scaler_d: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": _clean_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg.model_dump(),
        "extra": extra or {},
    }
    if disc is not None:
        payload["disc"] = _unwrap(disc).state_dict()
        payload["optimizer_d"] = optimizer_d.state_dict() if optimizer_d is not None else None
        payload["scaler_d"] = scaler_d.state_dict() if scaler_d is not None else None
    tmp = p.with_suffix(".tmp")
    torch.save(payload, str(tmp))
    tmp.rename(p)


def _global_local_jepa_loss(
    p_cat: torch.Tensor,
    local_mask_cat: torch.Tensor,
    num_globals: int,
    num_locals: int,
    lam_context: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """V-JEPA 2.1 dense loss with asymmetric global/local recipe (DINO-style).

    The per-frame center is computed from *globals only* — this is the "anchor".
    Heavily-masked local views don't pull the center toward zero. Then:
      L_global  : cross-global per-frame MSE, uniform mean over (g, b, t).
      L_predict : locals at masked frames, MSE against the global center.
      L_context : locals at visible frames, MSE weighted 1/sqrt(d_min) by
                  distance to the nearest masked frame in the same local view.

    p_cat:           (V*B, P, T'), V = num_globals + num_locals.
                     First num_globals*B rows are globals, rest are locals.
    local_mask_cat:  (num_locals*B, 1, T'), 1 at masked frames (waveform-derived).

    Returns (loss, l_global, l_predict, l_context) where the last three are
    detached scalars for logging.
    """
    G = int(num_globals)
    L = int(num_locals)
    V = G + L
    VB, D, T = p_cat.shape
    B = VB // V

    p = p_cat.view(V, B, D, T)
    p_g = p[:G]                                     # (G, B, D, T)
    p_l = p[G:]                                     # (L, B, D, T)

    center = p_g.mean(dim=0, keepdim=True)          # (1, B, D, T)  globals-only anchor

    # Cross-global consistency: pull each global to the anchor on every frame.
    err_g = (p_g - center).pow(2).mean(dim=2)       # (G, B, T)
    # Identically zero when num_globals == 1: the center IS the single global.
    l_global = err_g.mean()

    # Locals: predict + context against the globals-only center.
    err_l = (p_l - center).pow(2).mean(dim=2)       # (L, B, T)
    m = local_mask_cat.view(L, B, T).to(p.dtype)    # 1 on masked frames
    one_minus_m = 1.0 - m

    pred_num = (err_l * m).sum()
    pred_den = m.sum().clamp_min(1.0)
    l_predict = pred_num / pred_den

    big = float(T + 1)
    arange = torch.arange(T, device=err_l.device, dtype=err_l.dtype)  # (T,)
    is_masked = m > 0.5                                                # (L, B, T)
    # Forward: nearest masked index t' <= t. Indices outside masked positions are -big
    # so cummax keeps the most recent masked index; distance = t - that index.
    idx_fwd = torch.where(is_masked, arange.expand_as(m), torch.full_like(m, -big))
    recent_fwd = idx_fwd.cummax(dim=-1).values                         # (L, B, T)
    d_fwd = (arange - recent_fwd).clamp_min(0).clamp_max(big)
    # Where no masked frame yet seen, recent_fwd is -big and d_fwd would be huge.
    # clamp_max(big) keeps numerical sanity; downstream takes minimum with d_bwd.
    # Backward: nearest masked index t' >= t. cummin from the right.
    idx_bwd = torch.where(is_masked, arange.expand_as(m), torch.full_like(m, 2.0 * big))
    nearest_right = idx_bwd.flip(-1).cummin(dim=-1).values.flip(-1)    # (L, B, T)
    d_bwd = (nearest_right - arange).clamp_min(0).clamp_max(big)
    d_min = torch.minimum(d_fwd, d_bwd).clamp_min(1.0)
    ctx_w = one_minus_m / d_min.sqrt()

    ctx_num = (err_l * ctx_w).sum()
    ctx_den = ctx_w.sum().clamp_min(1e-6)
    l_context = ctx_num / ctx_den

    loss = l_global + l_predict + lam_context * l_context
    return loss, l_global.detach(), l_predict.detach(), l_context.detach()


def _participation_ratio(x: torch.Tensor) -> torch.Tensor:
    """Effective rank of x (N samples, D dims): (sum lambda)^2 / sum lambda^2
    of the covariance eigenvalues. Range [1, D]; ~D means isotropic.

    clamp_min(0): eigvalsh can emit tiny negative eigenvalues on
    near-singular covariances, pushing the ratio below its floor of 1.
    """
    xc = x - x.mean(dim=0)
    cov = (xc.T @ xc) / max(x.size(0) - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    return (eig.sum() ** 2) / (eig.pow(2).sum() + 1e-8)


def _decode(model: torch.nn.ModuleDict, z: torch.Tensor, target_len: int) -> torch.Tensor:
    return model["decoder"](z, target_len=target_len)


def main() -> None:
    # Flush stdout immediately so print() output appears over SSH/nohup/pipe.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    _shutdown = False

    def _stop_handler(sig: int, frame: object) -> None:
        nonlocal _shutdown
        print(f"\n[train] {signal.Signals(sig).name} received — stopping after current step.", flush=True)
        _shutdown = True

    # SIGTERM as well as SIGINT: Kaggle / Slurm / cloud preemption send SIGTERM
    # before the hard kill, so catching it lets us save last.pt on the way out.
    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    # Distributed (DDP) bootstrap. Launched via `torchrun --nproc_per_node=N`; with
    # no such env (plain `python train.py`) world_size==1, so every DDP-specific
    # path below is skipped and single-GPU behaviour is byte-for-byte unchanged.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    is_main = rank == 0
    if is_dist:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Hardware acceleration flags
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    # (Per-process VRAM cap is applied below, once cfg is loaded — it's tunable.)

    # Everything in the config is settable via trailing dotted overrides, e.g.
    #   python train.py --config configs/exp0.yaml train.max_steps=5000 train.log_interval_steps=50
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler with W&B")
    ap.add_argument("--profile_wait", type=int, default=0, help="Steps to wait before profiling")
    ap.add_argument("--profile_warmup", type=int, default=0, help="Steps to warm up profiler")
    ap.add_argument("--profile_active", type=int, default=1, help="Steps to actively profile")
    ap.add_argument(
        "--max_hours",
        type=float,
        default=None,
        help="Wall-clock budget in hours: stop cleanly and save last.pt after this "
        "long. For fixed-length sessions (e.g. Kaggle's 12h cap) set it below the "
        "limit (e.g. 11.5) so the final save lands before the hard kill.",
    )
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    cfg.resolved_config_path = args.config

    # Cap per-process VRAM now that cfg is known (must be after set_device, before
    # any large allocation — argparse/config load do none). Raise on a dedicated
    # GPU; leave ~0.9 GiB for the CUDA context + NCCL (outside PyTorch's allocator).
    torch.cuda.set_per_process_memory_fraction(cfg.run.gpu_mem_fraction, device=local_rank)

    # When resuming, pass run.run_id=<id> matching the checkpoint's run dir
    # (<out_dir>/<run_id>/checkpoints/<name>.pt) to reuse its out_dir and wandb run.

    # Per-rank seed offset: identical model init is enforced by DDP's broadcast at
    # wrap time anyway, so offsetting by rank only diversifies the per-rank waveform
    # augmentation / chunk-mask RNG. rank 0 -> unchanged from single-GPU.
    seed_all(cfg.run.seed + rank)

    run_id = cfg.run.run_id or _now_run_id()
    out_root = pathlib.Path(cfg.run.out_dir) / run_id
    ckpt_dir = out_root / "checkpoints"
    log_dir = out_root / "logs"

    # Only rank 0 touches disk / W&B; non-main ranks never create dirs or log.
    jsonl = None
    wb = None
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        jsonl = JsonlLogger(str(log_dir / "train.jsonl"))
        wb = maybe_init_wandb(cfg, run_id, str(out_root), resume=bool(args.resume))

    # Data
    dcfg = cfg.data
    if dcfg.train_manifest is None:
        raise ValueError("Set data.train_manifest to a JSONL manifest path (e.g. data/manifests/train.jsonl)")
    if is_main:
        meta_extra = {
            "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
            "train_manifest": str(dcfg.train_manifest),
            "val_manifest": str(dcfg.val_manifest or ""),
        }
        (out_root / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
        (out_root / "run_meta.yaml").write_text(yaml.safe_dump(meta_extra, sort_keys=False), encoding="utf-8")
    train_ds = AudioDataset(
        DatasetConfig(
            manifest=dcfg.train_manifest,
            sample_rate=dcfg.sample_rate,
            segment_seconds=dcfg.segment_seconds,
            random_crop=True,
        )
    )
    # DistributedSampler shards the data across ranks under DDP; set_epoch (in the
    # loop) reshuffles each pass. Single-GPU -> sampler None and shuffle=True (unchanged).
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        if is_dist
        else None
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        num_workers=dcfg.num_workers,
        pin_memory=dcfg.pin_memory,
        persistent_workers=dcfg.persistent_workers if dcfg.num_workers > 0 else False,
        # Deeper prefetch queue absorbs per-item decode/resample spikes (mp3 ->
        # 16k) so the GPU doesn't drain the queue between steps. Must be None when
        # there are no workers (torch rejects an int with num_workers=0).
        prefetch_factor=dcfg.prefetch_factor if dcfg.num_workers > 0 else None,
        # "spawn" avoids inheriting CUDA's locked mutexes from the parent process.
        # fork-after-CUDA-init deadlocks worker processes on Linux (the default).
        multiprocessing_context="spawn" if dcfg.num_workers > 0 else None,
        collate_fn=collate_fixed,
        drop_last=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )

    # Model
    mcfg = cfg.model
    frontend = ConvFrontend(mcfg.frontend)
    encoder = Encoder(frontend.out_channels, mcfg.encoder)

    latent_dim = mcfg.encoder.d_model

    decoder = WaveformDecoder(latent_dim, mcfg.decoder)

    projector = Projector(latent_dim, mcfg.projector)
    proj_dim = projector.output_dim

    sigreg = SIGReg(proj_dim, cfg.loss.sigreg)

    model = torch.nn.ModuleDict(
        {
            "frontend": frontend,
            "encoder": encoder,
            "projector": projector,
            "decoder": decoder,
            "sigreg": sigreg,
        }
    ).to(device)

    # One-time trainable-parameter breakdown (replaces scripts/get_param_count.py).
    _block_params = {
        name: sum(p.numel() for p in model[name].parameters() if p.requires_grad)
        for name in ("frontend", "encoder", "projector", "decoder", "sigreg")
    }
    if is_main:
        print("[train] trainable parameters:")
        for _name, _n in _block_params.items():
            print(f"  {_name:<10} {_n:>12,}")
        print(f"  {'total':<10} {sum(_block_params.values()):>12,}")

    # Adversarial discriminator (HiFi-GAN MPD). Built separately from `model`
    # so the generator optimizer / param breakdown stay clean; trained by its
    # own optimizer below. Disabled -> None and the loop runs exactly as before.
    acfg = cfg.loss.adv
    disc = None
    if acfg.enabled:
        disc = MultiPeriodDiscriminator(acfg.periods, channels=acfg.disc_channels).to(device)
        _d_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
        if is_main:
            print(f"  {'disc(MPD)':<10} {_d_params:>12,}  (adversarial, separate optimizer)")

    # Losses
    stft = MultiResSTFTLoss(cfg.loss.stft).to(device)
    wave_aug_cfg = cfg.aug.wave_aug
    wave_chunk_mask_cfg = cfg.aug.wave_chunk_mask

    # Frontend stride product gives samples-per-output-frame. Used to convert
    # frame-level chunk masks (local-view recipe) into waveform sample masks.
    samples_per_frame = math.prod(mcfg.frontend.strides)
    assert math.prod(mcfg.decoder.up_strides) == samples_per_frame, (
        f"decoder up_strides product {math.prod(mcfg.decoder.up_strides)} must equal "
        f"frontend strides product {samples_per_frame}"
    )
    # Pre-compute exact output frame count via one dummy frontend forward —
    # this is what we hand to make_frame_chunk_masks so the local-view mask
    # aligns 1:1 with the encoder grid.
    _seg_samples = int(round(dcfg.segment_seconds * dcfg.sample_rate))
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, _seg_samples, device=device)
        n_frames_per_segment = int(model["frontend"](_dummy).size(-1))

    # Optim
    ocfg = cfg.optim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ocfg.lr,
        betas=tuple(ocfg.betas),
        eps=ocfg.eps,
        weight_decay=ocfg.weight_decay,
    )

    scfg = ocfg.scheduler
    warmup_steps = scfg.warmup_steps
    total_steps = scfg.total_steps
    _cosine_inner = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=ocfg.lr * scfg.min_lr_ratio,
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

    use_amp = cfg.run.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Discriminator optimizer + scaler (constant LR; HiFi-GAN AdamW betas).
    optimizer_d = None
    scaler_d = None
    if disc is not None:
        optimizer_d = torch.optim.AdamW(
            disc.parameters(),
            lr=acfg.lr,
            betas=tuple(acfg.betas),
            eps=ocfg.eps,
            weight_decay=ocfg.weight_decay,
        )
        scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        if (state.get("extra") or {}).get("scheduler"):
            scheduler.load_state_dict(state["extra"]["scheduler"])
        if disc is not None and state.get("disc") is not None:
            disc.load_state_dict(state["disc"], strict=True)
            if state.get("optimizer_d") is not None:
                optimizer_d.load_state_dict(state["optimizer_d"])
            if state.get("scaler_d") and scaler_d is not None and scaler_d.is_enabled():
                scaler_d.load_state_dict(state["scaler_d"])
        step = int(state.get("step", 0))

    # Wrap param-bearing submodules in DDP AFTER any resume-load (so the clean
    # single-GPU checkpoint format loads into the raw modules). `sigreg` has no
    # learnable params, so it is never wrapped. DDP shares param tensors with the
    # raw modules, so the optimizer(s) built above stay valid. (If MHC ever leaves
    # encoder params unused in a forward, add find_unused_parameters=True there.)
    if is_dist:
        for _name in ("frontend", "encoder", "projector", "decoder"):
            model[_name] = torch.nn.parallel.DistributedDataParallel(
                model[_name], device_ids=[local_rank], output_device=local_rank
            )
        if disc is not None:
            disc = torch.nn.parallel.DistributedDataParallel(
                disc, device_ids=[local_rank], output_device=local_rank
            )

    # Training loop
    model.train()
    max_steps = cfg.train.max_steps
    grad_accum = cfg.train.grad_accum_steps
    grad_clip = cfg.optim.grad_clip

    jcfg = cfg.loss.jepa
    jepa_w = jcfg.weight
    num_globals = jcfg.num_globals
    num_locals = jcfg.num_locals
    if num_globals < 1 or num_locals < 1:
        raise ValueError(f"loss.jepa.num_globals and num_locals must both be >= 1; got G={num_globals}, L={num_locals}")
    sig_w = cfg.loss.sigreg.weight
    # SIGReg is gathered across ranks (see loop); DDP averages gradients by 1/W, so
    # scaling its term by world_size restores the single-GPU full-batch gradient.
    sig_scale = float(world_size)
    stft_w = cfg.loss.stft_weight
    wav_l1_w = cfg.loss.wav_l1_weight
    adv_w = acfg.adv_weight
    fm_w = acfg.fm_weight
    adv_start = acfg.adv_start_step
    fm_start = acfg.fm_start_step

    # V-JEPA 2.1 context-loss weight: relative weight of L_context vs L_predict
    # in the Dense Predictive Loss. Paper uses ~1.0 with distance weighting.
    lam_context_w = jcfg.context_weight

    def _extra_state(**extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"scheduler": scheduler.state_dict()}
        payload.update(extra)
        return payload

    prof = None
    if args.profile and is_main:
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

    accum_sums: Dict[str, torch.Tensor] = {}
    accum_counts: Dict[str, int] = {}

    start_time = time.monotonic()
    max_seconds = args.max_hours * 3600.0 if args.max_hours else None

    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_it = iter(train_dl)
    while step < max_steps:
        # Stop decision must be COLLECTIVE under DDP: if one rank breaks (SIGTERM or
        # wall-clock budget) while others keep going, the others hang at the next
        # collective. all_reduce(MAX) makes every rank stop on the same iteration.
        # (step < max_steps is already identical across ranks.)
        stop = _shutdown or (
            max_seconds is not None and time.monotonic() - start_time >= max_seconds
        )
        if is_dist:
            _stop_t = torch.tensor([1.0 if stop else 0.0], device=device)
            dist.all_reduce(_stop_t, op=dist.ReduceOp.MAX)
            stop = _stop_t.item() > 0.0
        if stop:
            if is_main:
                print(
                    f"[train] stopping at step {step} (shutdown or wall-clock budget) "
                    "— saving last.pt.",
                    flush=True,
                )
            break
        optimizer.zero_grad(set_to_none=True)
        if optimizer_d is not None:
            optimizer_d.zero_grad(set_to_none=True)

        total_loss = torch.tensor(0.0, device=device)

        microbatches = []
        for _ in range(grad_accum):
            try:
                batch = next(train_it)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                train_it = iter(train_dl)
                batch = next(train_it)
            microbatches.append(batch)

        for batch in microbatches:
            wav_a = batch["wav"]  # (B,1,T)

            wav_a = wav_a.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                B = wav_a.size(0)
                sr = dcfg.sample_rate

                # Globals: light wave aug, no chunk mask.
                view_wavs = [apply_waveform_augment(wav_a, sr, wave_aug_cfg) for _ in range(num_globals)]
                # Locals: wave aug + waveform-space chunk mask (sample-aligned to frames).
                local_frame_masks_cpu = make_frame_chunk_masks(
                    num_locals * B, n_frames_per_segment, wave_chunk_mask_cfg
                )
                local_frame_masks = local_frame_masks_cpu.to(device, non_blocking=True)
                for li in range(num_locals):
                    aug = apply_waveform_augment(wav_a, sr, wave_aug_cfg)
                    fmask = local_frame_masks[li * B:(li + 1) * B]   # (B, n_frames)
                    view_wavs.append(apply_waveform_chunk_mask(aug, fmask, samples_per_frame))
                wav_cat = torch.cat(view_wavs, dim=0)                 # (V*B, 1, T_wav)

                h0_cat = model["frontend"](wav_cat)
                # Locals carry the waveform chunk mask; globals get only the
                # light waveform augmentation (noise/lowpass/gain/clip), no
                # mask — that asymmetry is what makes the globals-only center
                # a usable anchor.
                z_cat = model["encoder"](h0_cat)                      # (V*B, D, T')
                p_cat = model["projector"](z_cat)                     # (V*B, P, T')

                local_mask_cat = local_frame_masks.unsqueeze(1)        # (L*B, 1, T')

                l_jepa, l_jepa_global_dbg, l_jepa_pred_dbg, l_jepa_ctx_dbg = _global_local_jepa_loss(
                    p_cat,
                    local_mask_cat,
                    num_globals=num_globals,
                    num_locals=num_locals,
                    lam_context=lam_context_w,
                )

                # ---- SIGReg (frame-level, on projector output only) ------------
                # Reshape p_cat (V*B, P, T') -> (T'*V*B, P) so
                # SIGReg treats each (frame, view, sample) triple as an
                # independent point. Paper N-scaling is left to SIGReg itself;
                # no extra /N rescaling here.
                D_lat = p_cat.size(1)
                sig_input = p_cat.permute(2, 0, 1)                # (T', V*B, P)
                sig_flat = sig_input.reshape(-1, D_lat)           # (T'*V*B, P)
                # Gather across ranks so the characteristic-function estimate (and
                # its ×N scaling) uses the GLOBAL batch, matching single-GPU. Only
                # the local slot carries grad; DDP + sig_scale (×W) restore the
                # single-GPU gradient magnitude.
                sig_global = _gather_with_grad(sig_flat, world_size, rank)
                l_sig = sigreg(sig_global, step=step)
                sig_stats = {
                    "l_sig_frm": l_sig.detach(),
                    "l_jepa_predict": l_jepa_pred_dbg,
                    "l_jepa_context": l_jepa_ctx_dbg,
                    "l_jepa_global": l_jepa_global_dbg,
                }

                # Diagnostic slicing: compare global-0 vs local-0 (clean vs masked signal).
                z_a = z_cat[:B]               # view-0 encoder embeddings (decoder + diagnostics)
                z_mask = z_cat[num_globals * B : (num_globals + 1) * B]

                # Decode from view-0 to clean wav_a (denoising reconstruction).
                x_hat = _decode(model, z_a, target_len=wav_a.size(-1))

                l_stft_ps, stft_stats_ps = stft(x_hat, wav_a, return_per_sample=True)
                l_stft = l_stft_ps.mean()
                stft_stats = {k: v.mean().detach() for k, v in stft_stats_ps.items()}

                l_wav_ps = (x_hat - wav_a).abs().mean(dim=(1, 2))
                l_wav = l_wav_ps.mean()

                # ---- Discriminator update (MPD) -------------------------------
                # The whole GAN path is skipped until step >= adv_start: no point
                # training D before the generator uses its signal, and skipping it
                # keeps the pre-GAN phase fast / low-VRAM. Real + detached fake ->
                # this graph only touches D; requires_grad True so the D backward
                # (below) accumulates D grads. (Assumes adv_start <= fm_start.)
                disc_active = disc is not None and step >= adv_start
                loss_d = None
                if disc_active:
                    for p in disc.parameters():
                        p.requires_grad_(True)
                    d_real, d_fake, _, _ = disc(wav_a, x_hat.detach())
                    loss_d = discriminator_loss(d_real, d_fake)

            if loss_d is not None:
                scaler_d.scale(loss_d / grad_accum).backward()

            with torch.amp.autocast("cuda", enabled=use_amp):
                # ---- Generator adversarial + feature matching ----------------
                # Freeze D params: gradient still flows THROUGH D into x_hat (the
                # generator signal), but D grads aren't accumulated, so the G
                # backward can't corrupt the D grads accumulated above (needed
                # because we accumulate over microbatches before stepping).
                l_adv = x_hat.new_zeros(())
                l_fm = x_hat.new_zeros(())
                if disc_active:
                    for p in disc.parameters():
                        p.requires_grad_(False)
                    _, g_fake, fmap_r, fmap_g = _unwrap(disc)(wav_a, x_hat)
                    l_adv = generator_adv_loss(g_fake)
                    if step >= fm_start:
                        l_fm = feature_matching_loss(fmap_r, fmap_g)

                loss = (
                    stft_w * l_stft
                    + wav_l1_w * l_wav
                    + jepa_w * l_jepa
                    + sig_w * sig_scale * l_sig
                    + adv_w * l_adv
                    + fm_w * l_fm
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            total_loss = total_loss + loss.detach()

            # JEPA collapse detector on z — cheap, run every microbatch. z is
            # what the decoder and downstream probes consume, so monitor it
            # directly rather than the projector output. Raw RMS values are
            # pinned ~1 by LayerNorm and carry no signal; the informative
            # quantity is diff relative to norm.
            # (Rank gauges are computed at log boundaries below, NOT here:
            # accumulating 0.0 placeholders on non-computed steps silently
            # divided every logged rank by log_interval.)
            with torch.no_grad():
                z_a_rms = z_a.pow(2).mean().sqrt()
                z_diff_rms = (z_a - z_mask).pow(2).mean().sqrt()
                z_to_norm_ratio = z_diff_rms / z_a_rms.clamp_min(1e-6)

            mb_step_stats = {
                "l_stft": l_stft.detach(),
                "l_wav": l_wav.detach(),
                "l_jepa": l_jepa.detach(),
                "z_diff_rms": z_diff_rms.detach(),
                "z_to_norm_ratio": z_to_norm_ratio.detach(),
            }
            if disc is not None:
                mb_step_stats["l_adv"] = l_adv.detach()
                mb_step_stats["l_fm"] = l_fm.detach()
                if loss_d is not None:
                    mb_step_stats["l_disc"] = loss_d.detach()
            mb_step_stats.update({k: v.detach() for k, v in stft_stats.items()})
            mb_step_stats.update({k: v.detach() for k, v in sig_stats.items()})

            for k, v in mb_step_stats.items():
                accum_sums[k] = accum_sums.get(k, torch.zeros((), device=device)) + v.detach()
                accum_counts[k] = accum_counts.get(k, 0) + 1

        accum_sums["loss"] = accum_sums.get("loss", torch.zeros((), device=device)) + total_loss.detach()
        accum_counts["loss"] = accum_counts.get("loss", 0) + 1

        # Optimize generator
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Optimize discriminator (only once the GAN path is active)
        if optimizer_d is not None and step >= adv_start:
            if grad_clip and grad_clip > 0:
                scaler_d.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
            scaler_d.step(optimizer_d)
            scaler_d.update()

        step += 1
        scheduler.step()

        # Embedding similarity probe: encode the last microbatch clean and
        # augmented (global-recipe aug, eval mode, no grad), then compare
        # same-utterance pairs against shifted (different-utterance) pairs at
        # frame and utterance level. pos ≪ neg means the encoder identifies
        # the same audio under augmentation while keeping utterances apart;
        # contrast (neg/pos) drifting toward 1 means collapse or aug-sensitivity.
        if is_main and step % cfg.train.probe_interval_steps == 0:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                z_clean = _unwrap(model["encoder"])(_unwrap(model["frontend"])(wav_a)).float()
                z_aug = _unwrap(model["encoder"])(
                    _unwrap(model["frontend"])(apply_waveform_augment(wav_a, dcfg.sample_rate, wave_aug_cfg))
                ).float()
                pos_frame = (z_clean - z_aug).pow(2).mean()
                neg_frame = (z_clean - z_aug.roll(1, dims=0)).pow(2).mean()
                zc_utt = z_clean.mean(dim=2)
                za_utt = z_aug.mean(dim=2)
                pos_utt = (zc_utt - za_utt).pow(2).mean()
                neg_utt = (zc_utt - za_utt.roll(1, dims=0)).pow(2).mean()
            model.train()
            probe_row = {
                "step": step,
                "sim/pos_frame_mse": pos_frame.item(),
                "sim/neg_frame_mse": neg_frame.item(),
                "sim/frame_contrast": (neg_frame / pos_frame.clamp_min(1e-8)).item(),
                "sim/pos_utt_mse": pos_utt.item(),
                "sim/neg_utt_mse": neg_utt.item(),
                "sim/utt_contrast": (neg_utt / pos_utt.clamp_min(1e-8)).item(),
            }
            jsonl.log(probe_row)
            if wb is not None:
                wb.log(probe_row, step=step)

        log_interval = cfg.train.log_interval_steps
        if is_main and step % log_interval == 0:
            log_stats = {k: (v / accum_counts[k]).item() for k, v in accum_sums.items()}
            accum_sums.clear()
            accum_counts.clear()
            # Rank gauges: eigendecompositions are expensive, so compute them
            # once per log boundary from the last microbatch's view-0
            # embeddings and log the raw value — these must NOT pass through
            # accum_sums, whose per-key averaging diluted them 10x.
            with torch.no_grad():
                z32 = z_a.detach().float()
                z_frames = z32.permute(0, 2, 1).reshape(-1, z32.size(1))
                z_res = (z32 - z32.mean(dim=2, keepdim=True)).permute(0, 2, 1).reshape(-1, z32.size(1))
                log_stats["z_rank"] = _participation_ratio(z_frames).item()
                log_stats["z_rank_utt"] = _participation_ratio(z32.mean(dim=2)).item()
                log_stats["z_rank_res"] = _participation_ratio(z_res).item()
                # Pooled-p rank: the utterance-level SIGReg term was cut, so
                # watch this gauge — if it sags toward 1-2, pooled embeddings
                # are collapsing and the term should come back.
                log_stats["p_rank_utt"] = _participation_ratio(p_cat[:B].detach().float().mean(dim=2)).item()
            row = {"step": step, **log_stats}

            encoder_mod = _unwrap(model["encoder"])
            if hasattr(encoder_mod, "mhc_wrappers"):
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

        if is_main and step % cfg.train.save_interval_steps == 0:
            # Step-tagged so a later collapse still leaves usable checkpoints
            # to roll back to / post-mortem; last.pt stays the resume target.
            step_ckpt = ckpt_dir / f"step_{step:06d}.pt"
            save_checkpoint(
                str(step_ckpt),
                step=step,
                model=model,
                optimizer=optimizer,
                scaler=scaler if scaler.is_enabled() else None,
                cfg=cfg,
                extra=_extra_state(),
                disc=disc,
                optimizer_d=optimizer_d,
                scaler_d=scaler_d if (scaler_d is not None and scaler_d.is_enabled()) else None,
            )
            shutil.copyfile(step_ckpt, ckpt_dir / "last.pt")

        if prof is not None:
            prof.step()

    if prof is not None:
        prof.stop()

    if is_main:
        save_checkpoint(
            str(ckpt_dir / "last.pt"),
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=scaler if scaler.is_enabled() else None,
            cfg=cfg,
            extra=_extra_state(),
            disc=disc,
            optimizer_d=optimizer_d,
            scaler_d=scaler_d if (scaler_d is not None and scaler_d.is_enabled()) else None,
        )

    # Barrier so non-main ranks wait for rank 0's final save before tearing the
    # process group down (destroy can race a still-writing rank 0 otherwise).
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
