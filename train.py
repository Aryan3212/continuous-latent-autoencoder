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
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

# Direct training commands automatically read local secrets/config. Shell
# exports still win, which is important on managed training platforms.
load_dotenv(pathlib.Path(__file__).resolve().parent / ".env", override=False)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist

from data_loading import (
    apply_frame_mask,
    apply_waveform_augment,
    apply_waveform_chunk_mask,
    make_span_masks,
)
from data_loading import (
    AudioDataset,
    DatasetConfig,
    PackedTarDataset,
    collate_fixed,
    packed_worker_init,
)
from losses import (
    MelLoss,
    MultiResSTFTLoss,
    ReconSpectrogram,
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
)
from models.autoencoder import Autoencoder
from models.discriminator import MultiPeriodDiscriminator
from models.sigreg import SIGReg
from models.visreg import VISReg
from config import apply_overrides, load_config
from schema import Config


@dataclass(frozen=True)
class _ScheduleInputs:
    lr: float
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float


def _schedule_inputs(cfg: Config) -> _ScheduleInputs:
    scheduler = cfg.optim.scheduler
    return _ScheduleInputs(
        lr=cfg.optim.lr,
        warmup_steps=scheduler.warmup_steps,
        total_steps=scheduler.total_steps or cfg.train.max_steps,
        min_lr_ratio=scheduler.min_lr_ratio,
    )


def _checkpoint_schedule_inputs(state: dict[str, Any]) -> _ScheduleInputs | None:
    try:
        cfg = state["cfg"]
        optim = cfg["optim"]
        scheduler = optim["scheduler"]
        total_steps = scheduler["total_steps"] or cfg["train"]["max_steps"]
        return _ScheduleInputs(
            lr=float(optim["lr"]),
            warmup_steps=int(scheduler["warmup_steps"]),
            total_steps=int(total_steps),
            min_lr_ratio=float(scheduler["min_lr_ratio"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _warn_schedule_change(saved: _ScheduleInputs, current: _ScheduleInputs) -> None:
    labels = {
        "lr": "optim.lr",
        "warmup_steps": "optim.scheduler.warmup_steps",
        "total_steps": "effective total_steps",
        "min_lr_ratio": "optim.scheduler.min_lr_ratio",
    }
    changes = [
        f"{labels[name]}: checkpoint={getattr(saved, name)}, current={getattr(current, name)}"
        for name in labels
        if getattr(saved, name) != getattr(current, name)
    ]
    if changes:
        print(
            "[train] WARNING: LR schedule inputs changed on resume; using the current "
            f"schedule ({'; '.join(changes)}).",
            flush=True,
        )


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


def _clean_state_dict(model: torch.nn.Module) -> dict[str, Any]:
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

    def log(self, row: dict[str, Any]) -> None:
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
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: Config,
    disc: torch.nn.Module | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    scaler_d: torch.cuda.amp.GradScaler | None = None,
) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": _clean_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg.model_dump(),
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
    num_globals: int,
    num_locals: int,
) -> torch.Tensor:
    """Simplified JEPA loss: uniform MSE on all frames.

    Asymmetric global/local recipe (DINO-style):
      - The per-frame center is computed from globals only (anchor).
      - All local frames are pulled toward this center uniformly.

    p_cat: (V*B, P, T'), V = num_globals + num_locals.
           First num_globals*B rows are globals, rest are locals.
    """
    G = int(num_globals)
    L = int(num_locals)
    V = G + L
    VB, D, T = p_cat.shape
    B = VB // V

    p = p_cat.view(V, B, D, T)
    p_g = p[:G]
    p_l = p[G:]

    center = p_g.mean(dim=0, keepdim=True)

    l_global = (p_g - center).pow(2).mean()
    l_jepa = (p_l - center).pow(2).mean()

    return l_global + l_jepa


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler with W&B")
    parser.add_argument("--profile_wait", type=int, default=0, help="Steps to wait before profiling")
    parser.add_argument("--profile_warmup", type=int, default=0, help="Steps to warm up profiler")
    parser.add_argument("--profile_active", type=int, default=1, help="Steps to actively profile")
    parser.add_argument(
        "--max_hours",
        type=float,
        default=None,
        help="Stop cleanly after this many hours and save last.pt.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
    completed_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    decay_steps = max(1, total_steps - warmup_steps)
    warmup_start = 1.0 / max(1, warmup_steps)

    def lr_ratio(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return warmup_start + (1.0 - warmup_start) * step / warmup_steps
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    for group in optimizer.param_groups:
        group["initial_lr"] = base_lr
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_ratio,
        last_epoch=completed_steps - 1,
    )


def _reduce_metric_totals(
    sums: dict[str, torch.Tensor],
    counts: dict[str, int],
    *,
    distributed: bool,
    device: torch.device,
) -> dict[str, float]:
    keys = sorted(sums)
    sum_values = torch.stack([sums[key] for key in keys])
    count_values = torch.tensor(
        [counts[key] for key in keys], device=device, dtype=sum_values.dtype
    )
    if distributed:
        dist.all_reduce(sum_values)
        dist.all_reduce(count_values)
    return {
        key: (value / count).item()
        for key, value, count in zip(keys, sum_values, count_values)
    }


def _all_gather_batch(x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return x
    x = x.contiguous()
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)


def _restore_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    disc: torch.nn.Module | None,
    optimizer_d: torch.optim.Optimizer | None,
    scaler_d: torch.cuda.amp.GradScaler | None,
) -> tuple[int, _ScheduleInputs | None]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if state.get("scaler") and scaler.is_enabled():
        scaler.load_state_dict(state["scaler"])
    if disc is not None and state.get("disc") is not None:
        disc.load_state_dict(state["disc"], strict=True)
        if state.get("optimizer_d") is not None:
            optimizer_d.load_state_dict(state["optimizer_d"])
        if state.get("scaler_d") and scaler_d is not None and scaler_d.is_enabled():
            scaler_d.load_state_dict(state["scaler_d"])
    return int(state.get("step", 0)), _checkpoint_schedule_inputs(state)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    _shutdown = False

    def _stop_handler(sig: int, frame: object) -> None:
        nonlocal _shutdown
        print(f"\n[train] {signal.Signals(sig).name} received — stopping after current step.", flush=True)
        _shutdown = True

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

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    args = _parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    _AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = _AMP_DTYPES[cfg.run.amp_dtype]

    torch.cuda.set_per_process_memory_fraction(cfg.run.gpu_mem_fraction, device=local_rank)

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
    if is_main:
        meta_extra = {
            "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
            "data_backend": dcfg.backend,
            "train_manifest": str(dcfg.train_manifest),
            "shard_manifest": str(dcfg.shard_manifest or ""),
            "val_manifest": str(dcfg.val_manifest or ""),
        }
        (out_root / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
        (out_root / "run_meta.yaml").write_text(yaml.safe_dump(meta_extra, sort_keys=False), encoding="utf-8")
    if dcfg.backend == "files":
        train_ds: torch.utils.data.Dataset[dict[str, Any]] | torch.utils.data.IterableDataset[dict[str, Any]] = AudioDataset(
            DatasetConfig(
                manifest=dcfg.train_manifest,
                sample_rate=dcfg.sample_rate,
                segment_seconds=dcfg.segment_seconds,
                random_crop=True,
            )
        )
        # DistributedSampler shards the data across ranks under DDP; set_epoch (in
        # the loop) reshuffles each pass. Single-GPU -> shuffle=True (unchanged).
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
            )
            if is_dist
            else None
        )
        train_shuffle = train_sampler is None
        worker_init_fn = None
    else:
        train_ds = PackedTarDataset(
            shard_manifest=str(dcfg.shard_manifest),
            sample_rate=dcfg.sample_rate,
            segment_seconds=dcfg.segment_seconds,
            random_crop=True,
            shuffle_buffer_mb=dcfg.shuffle_buffer_mb,
            run_seed=cfg.run.seed,
            rank=rank,
            world_size=world_size,
            workers_per_rank=dcfg.num_workers,
            batch_size=cfg.train.batch_size,
        )
        train_sampler = None
        train_shuffle = False
        worker_init_fn = packed_worker_init
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
        shuffle=train_shuffle,
        worker_init_fn=worker_init_fn,
    )

    # Model
    mcfg = cfg.model
    model = Autoencoder(mcfg)
    # Gaussianisation loss: SIGReg (sliced characteristic-function test) or VISReg
    # (vector isotropic Gaussianisation). Both are param-free and act frame-level
    # on the projector output; selected by loss.reg_type. Stored under the
    # reg_type key so resuming an existing sigreg checkpoint (which saved the
    # param-free module under "sigreg") still loads under strict=True.
    reg_type = cfg.loss.reg_type
    if reg_type == "sigreg":
        reg = SIGReg(cfg.loss.sigreg)
    else:
        reg = VISReg(cfg.loss.visreg)

    model.add_module(reg_type, reg)
    model.to(device)

    # One-time trainable-parameter breakdown (replaces scripts/get_param_count.py).
    _block_params = {
        name: sum(p.numel() for p in getattr(model, name).parameters() if p.requires_grad)
        for name in ("frontend", "encoder", "projector", "decoder", reg_type)
    }
    if is_main:
        print("[train] trainable parameters:")
        for _name, _n in _block_params.items():
            print(f"  {_name:<10} {_n:>12,}")
        print(f"  {'total':<10} {sum(_block_params.values()):>12,}")

    recon_type = cfg.loss.recon_type
    if recon_type == "mel":
        recon_fn = MelLoss(cfg.loss.mel, sample_rate=dcfg.sample_rate).to(device)
        stft_fn = MultiResSTFTLoss(cfg.loss.stft).to(device)
    else:
        recon_fn = MultiResSTFTLoss(cfg.loss.stft).to(device)
        stft_fn = recon_fn

    acfg = cfg.loss.adv
    disc = None
    disc_spec = None
    if acfg.enabled:
        disc_spec = ReconSpectrogram(cfg.loss, dcfg.sample_rate).to(device)
        disc = MultiPeriodDiscriminator(
            acfg.periods, channels=acfg.disc_channels, in_channels=disc_spec.n_bins
        ).to(device)
        _d_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
        if is_main:
            print(f"  {'disc(MPD)':<10} {_d_params:>12,}  (adversarial, separate optimizer; "
                  f"input = {recon_type} spectrogram, {disc_spec.n_bins} bins)")

    aug_global_cfg = cfg.aug.waveform_aug_global
    aug_local_cfg = cfg.aug.waveform_aug_local or aug_global_cfg
    wave_mask_cfg = cfg.aug.waveform_aug_local_mask
    frontend_mask_cfg = cfg.aug.frontend_frame_local_mask
    frontend_noise_cfg = cfg.aug.frontend_frame_noise
    decoder_mask_cfg = cfg.aug.decoder_input_mask
    decoder_noise_cfg = cfg.aug.decoder_input_noise

    # Frontend stride product gives samples-per-output-frame. Used to convert
    # frame-level chunk masks (local-view recipe) into waveform sample masks.
    samples_per_frame = math.prod(mcfg.frontend.strides)
    # Pre-compute exact output frame count via one dummy frontend forward —
    # this is what we hand to make_span_masks so the local-view mask
    # aligns 1:1 with the encoder grid.
    _seg_samples = int(math.ceil(dcfg.segment_seconds * dcfg.sample_rate))
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, _seg_samples, device=device)
        n_frames_per_segment = int(model.frontend(_dummy).size(-1))

    # Optim
    ocfg = cfg.optim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ocfg.lr,
        betas=tuple(ocfg.betas),
        eps=ocfg.eps,
        weight_decay=ocfg.weight_decay,
    )

    schedule_inputs = _schedule_inputs(cfg)

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
    saved_schedule_inputs: _ScheduleInputs | None = None
    if args.resume:
        step, saved_schedule_inputs = _restore_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            disc=disc,
            optimizer_d=optimizer_d,
            scaler_d=scaler_d,
        )
        if is_main and saved_schedule_inputs is not None:
            _warn_schedule_change(saved_schedule_inputs, schedule_inputs)

    scheduler = _build_scheduler(
        optimizer,
        base_lr=schedule_inputs.lr,
        warmup_steps=schedule_inputs.warmup_steps,
        total_steps=schedule_inputs.total_steps,
        min_lr_ratio=schedule_inputs.min_lr_ratio,
        completed_steps=step,
    )
    # Wrap param-bearing submodules in DDP AFTER any resume-load (so the clean
    # single-GPU checkpoint format loads into the raw modules). The Gaussianisation
    # module (sigreg/visreg) has no learnable params, so it is never wrapped. DDP
    # shares param tensors with the raw modules, so the optimizer(s) built above
    # stay valid. (If MHC ever leaves encoder params unused in a forward, add
    # find_unused_parameters=True there.)
    if is_dist:
        for _name in ("frontend", "encoder", "projector", "decoder"):
            module = getattr(model, _name)
            setattr(model, _name, torch.nn.parallel.DistributedDataParallel(
                module, device_ids=[local_rank], output_device=local_rank
            ))
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
    num_views = num_globals + num_locals
    reg_w = cfg.loss.sigreg.weight if reg_type == "sigreg" else cfg.loss.visreg.weight
    # The Gaussianisation loss is gathered across ranks (see loop); DDP averages
    # gradients by 1/W, so scaling its term by world_size restores the single-GPU
    # full-batch gradient.
    reg_scale = float(world_size)
    recon_w = cfg.loss.recon_weight
    recon_views = cfg.loss.recon_views
    recon_log_start = cfg.loss.recon_log_start_step
    adv_w = acfg.adv_weight
    fm_w = acfg.fm_weight
    adv_start = acfg.adv_start_step
    fm_start = acfg.fm_start_step
    adaptive_adv = acfg.adaptive
    adaptive_max = acfg.adaptive_max

    prof = None
    if args.profile and is_main:
        import wandb
        if wandb.run is None:
            print("Warning: W&B is not initialized, logging profiler trace to local './profiler_logs' directory.")
            handler = torch.profiler.tensorboard_trace_handler('./profiler_logs')
        else:
            handler = wandb.profiler.torch_trace_handler()

        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=args.profile_wait,
                warmup=args.profile_warmup,
                active=args.profile_active,
                repeat=1,
            ),
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

    accum_sums: dict[str, torch.Tensor] = {}
    accum_counts: dict[str, int] = {}

    start_time = time.monotonic()
    max_seconds = args.max_hours * 3600.0 if args.max_hours else None

    epoch = 0
    # Packed resume intentionally starts a fresh randomized data epoch instead
    # of trying to reconstruct the sample after the checkpoint. The completed
    # optimizer step is already checkpointed and supplies a deterministic seed.
    data_epoch = max(0, step) if dcfg.backend == "tar" else 0
    if dcfg.backend == "tar":
        assert isinstance(train_ds, PackedTarDataset)
        train_ds.set_epoch(data_epoch)
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

        # Adaptive adversarial weight is computed ONCE per optimizer step (on the
        # first disc-active microbatch) and reused across the accumulation window:
        # it's a slowly-varying scalar, and recomputing it every microbatch would
        # add two extra partial backward passes per microbatch.
        lam_adv_cached: torch.Tensor | None = None

        for _ in range(grad_accum):
            try:
                batch = next(train_it)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                if dcfg.backend == "tar":
                    data_epoch += 1
                    assert isinstance(train_ds, PackedTarDataset)
                    train_ds.set_epoch(data_epoch)
                train_it = iter(train_dl)
                batch = next(train_it)
            wav_a = batch["wav"]  # (B,1,T)

            wav_a = wav_a.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                B = wav_a.size(0)
                sr = dcfg.sample_rate

                view_wavs = [
                    apply_waveform_augment(wav_a, sr, aug_global_cfg)
                    for _ in range(num_globals)
                ]
                wave_masks = make_span_masks(
                    num_locals * B, n_frames_per_segment, wave_mask_cfg
                ).to(device, non_blocking=True)
                for li in range(num_locals):
                    aug = apply_waveform_augment(wav_a, sr, aug_local_cfg)
                    chunk_mask = wave_masks[li * B:(li + 1) * B]
                    view_wavs.append(
                        apply_waveform_chunk_mask(aug, chunk_mask, samples_per_frame)
                    )
                wav_cat = torch.cat(view_wavs, dim=0)                 # (V*B, 1, T_wav)

                h0_cat = model.frontend(wav_cat)
                g = num_globals * B
                n_local = num_locals * B
                if frontend_mask_cfg.enabled or frontend_noise_cfg.enabled:
                    local_h0 = h0_cat[g:]
                    if frontend_mask_cfg.enabled:
                        feature_masks = make_span_masks(
                            n_local, h0_cat.size(-1), frontend_mask_cfg
                        )
                        local_h0 = apply_frame_mask(local_h0, feature_masks)
                    if frontend_noise_cfg.enabled:
                        local_h0 = local_h0 + frontend_noise_cfg.std * torch.randn_like(
                            local_h0
                        )
                    h0_cat = torch.cat((h0_cat[:g], local_h0), dim=0)
                z_cat = model.encoder(h0_cat)                         # (V*B, D, T')
                p_cat = model.projector(z_cat)                        # (V*B, P, T')

                l_jepa = _global_local_jepa_loss(
                    p_cat,
                    num_globals=num_globals,
                    num_locals=num_locals,
                )

                # ---- Gaussianisation loss (frame-level, on projector output) ---
                # Reshape p_cat (V*B, P, T') -> (T'*V*B, P) so the loss treats
                # each (frame, view, sample) triple as an independent point.
                D_lat = p_cat.size(1)
                sig_input = p_cat.permute(2, 0, 1)                # (T', V*B, P)
                sig_flat = sig_input.reshape(-1, D_lat)           # (T'*V*B, P)
                # Gather across ranks so the estimate (and its ×N scaling) uses
                # the GLOBAL batch, matching single-GPU. Only the local slot
                # carries grad; DDP + reg_scale (×W) restore the single-GPU
                # gradient magnitude.
                sig_global = _gather_with_grad(sig_flat, world_size, rank)
                if reg_type == "sigreg":
                    l_reg = reg(sig_global, step=step)
                    reg_stats = {"l_sig": l_reg.detach()}
                else:  # visreg
                    # VISReg expects (N, B, D); treat the whole global batch as a
                    # single population (N=1, B=T'*V*B) so the Gaussianity target
                    # uses the largest possible pool of points.
                    l_reg = reg(sig_global.unsqueeze(0))
                    reg_stats = {"l_vis": l_reg.detach()}

                # Diagnostic slicing: compare global-0 vs local-0 (clean vs masked signal).
                z_a = z_cat[:B]               # view-0 encoder embeddings (decoder + diagnostics)
                z_mask = z_cat[num_globals * B : (num_globals + 1) * B]

                if recon_views == "global":
                    z_dec = z_a
                    clean_targets = wav_a
                elif recon_views == "local":
                    z_dec = z_cat[g:]
                    clean_targets = wav_a.repeat(num_locals, 1, 1)
                else:  # all
                    z_dec = z_cat
                    clean_targets = wav_a.repeat(num_views, 1, 1)

                # The decoder corruptions apply to every selected reconstruction
                # view. This keeps global-only, local-only, and all-view ablations
                # comparable; selection controls only which views are decoded.
                if decoder_mask_cfg.enabled:
                    decoder_masks = make_span_masks(
                        z_dec.size(0), z_dec.size(-1), decoder_mask_cfg
                    )
                    z_dec = apply_frame_mask(z_dec, decoder_masks)
                if decoder_noise_cfg.enabled:
                    z_dec = z_dec + decoder_noise_cfg.std * torch.randn_like(z_dec)

                x_hat_all = model.decoder(z_dec, target_len=wav_a.size(-1))
                x_hat = x_hat_all[:B]

                # Active reconstruction loss (training signal). Returns
                # stats prefixed stft_* or mel_* depending on recon_type.
                l_recon_ps, recon_stats_ps = recon_fn(
                    x_hat_all, clean_targets, return_per_sample=True
                )
                l_recon = l_recon_ps.mean()
                recon_stats = {k: v.mean().detach() for k, v in recon_stats_ps.items()}

                l_wav_ps = (x_hat_all - clean_targets).abs().mean(dim=(1, 2))
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
                    # Spectrograms (real + generated) computed ONCE and reused for
                    # both the discriminator update and the generator/feature-matching
                    # block below — avoids recomputing disc_spec twice per step.
                    for p in disc.parameters():
                        p.requires_grad_(True)
                    spec_real = disc_spec(wav_a)
                    spec_fake = disc_spec(x_hat)
                    d_real, d_fake, _, _ = disc(spec_real, spec_fake.detach())
                    loss_d = discriminator_loss(d_real, d_fake, acfg.loss_type)

            if loss_d is not None:
                scaler_d.scale(loss_d / grad_accum).backward()

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
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
                    _, g_fake, fmap_r, fmap_g = _unwrap(disc)(spec_real, spec_fake)
                    l_adv = generator_adv_loss(g_fake, acfg.loss_type)
                    if step >= fm_start:
                        l_fm = feature_matching_loss(fmap_r, fmap_g)

                # ---- Adaptive adversarial weight (VQGAN-style) ----------------
                # lambda balances the adversarial gradient to parity with the
                # reconstruction gradient at the decoder's last conv, so adv_w is a
                # clean relative-strength knob. Computed once per step (cached over
                # the accumulation window). autograd.grad reads grads w.r.t. last_w
                # directly without firing DDP's reduce hooks, so the per-rank
                # reducer still sees only the loss.backward() below. The gs prescale
                # cancels in the ratio but keeps fp16 grads off the AMP underflow
                # floor; clamp + eps bound a vanishing-adv-grad blowup (grad_clip
                # is the final backstop).
                lam_adv = x_hat.new_ones(())
                if disc_active and adaptive_adv:
                    if lam_adv_cached is None:
                        last_w = _unwrap(model.decoder).out_conv.weight
                        # gs lifts grads off the fp16 underflow floor; it cancels in
                        # the ratio so its value is arbitrary. Kept modest (not 1e3)
                        # so gs*l_adv can't overflow fp16 to inf -> inf/inf = NaN.
                        gs = 1.0e1
                        rec_g = torch.autograd.grad(
                            gs * (recon_w * l_recon), last_w, retain_graph=True
                        )[0]
                        adv_g = torch.autograd.grad(gs * l_adv, last_w, retain_graph=True)[0]
                        lam_adv_cached = (
                            rec_g.float().norm() / (adv_g.float().norm() + gs * 1e-3)
                        ).clamp(0.0, adaptive_max)
                        # clamp bounds inf but passes NaN through (inf/inf at GAN
                        # onset when adv_g underflows); nan->0 is the load-bearing guard.
                        lam_adv_cached = torch.nan_to_num(
                            lam_adv_cached, nan=0.0, posinf=adaptive_max
                        ).detach()
                    lam_adv = lam_adv_cached

                loss = (
                    recon_w * l_recon
                    + jepa_w * l_jepa
                    + reg_w * reg_scale * l_reg
                    + adv_w * lam_adv * l_adv
                    + fm_w * l_fm
                )
                loss = loss / grad_accum

                reported_loss = (
                    recon_w * l_recon.detach()
                    + jepa_w * l_jepa.detach()
                    + reg_w * l_reg.detach()
                    + adv_w * lam_adv.detach() * l_adv.detach()
                    + fm_w * l_fm.detach()
                ) / grad_accum

            scaler.scale(loss).backward()
            total_loss = total_loss + reported_loss

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
                ("l_stft" if recon_type == "stft" else "l_mel"): l_recon.detach(),
                "l_wav": l_wav.detach(),
                "l_jepa": l_jepa.detach(),
                "z_diff_rms": z_diff_rms.detach(),
                "z_to_norm_ratio": z_to_norm_ratio.detach(),
            }
            if disc is not None:
                mb_step_stats["l_adv"] = l_adv.detach()
                mb_step_stats["l_fm"] = l_fm.detach()
                if disc_active and adaptive_adv:
                    mb_step_stats["lam_adv"] = lam_adv.detach()
                if loss_d is not None:
                    mb_step_stats["l_disc"] = loss_d.detach()

            # Active recon breakdown (stft_* or mel_*) — primary diagnostic.
            if recon_type == "mel":
                mb_step_stats.update({k: v.detach() for k, v in recon_stats.items()})

            # STFT log-magnitude metric (ablation comparison), logged in BOTH recon
            # modes but only after recon_log_start_step, so early noisy steps stay
            # out of the curves.
            if step >= recon_log_start:
                if recon_type == "mel":
                    _, stft_ps = stft_fn(x_hat, wav_a, return_per_sample=True)
                    stft_stats = {k: v.mean().detach() for k, v in stft_ps.items()}
                else:
                    stft_stats = recon_stats
                mb_step_stats.update({k: v.detach() for k, v in stft_stats.items()})
            mb_step_stats.update({k: v.detach() for k, v in reg_stats.items()})

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
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                z_clean = _unwrap(model.encoder)(_unwrap(model.frontend)(wav_a)).float()
                z_aug = _unwrap(model.encoder)(
                    _unwrap(model.frontend)(apply_waveform_augment(wav_a, dcfg.sample_rate, aug_global_cfg))
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
        if step % log_interval == 0:
            log_stats = _reduce_metric_totals(
                accum_sums,
                accum_counts,
                distributed=is_dist,
                device=device,
            )
            accum_sums.clear()
            accum_counts.clear()

            with torch.no_grad():
                z32 = _all_gather_batch(z_a.detach().float(), world_size)
                p_utt = _all_gather_batch(
                    p_cat[:B].detach().float().mean(dim=2), world_size
                )

            if is_main:
                with torch.no_grad():
                    z_frames = z32.permute(0, 2, 1).reshape(-1, z32.size(1))
                    z_res = (z32 - z32.mean(dim=2, keepdim=True)).permute(0, 2, 1).reshape(-1, z32.size(1))
                    log_stats["z_rank"] = _participation_ratio(z_frames).item()
                    log_stats["z_rank_utt"] = _participation_ratio(z32.mean(dim=2)).item()
                    log_stats["z_rank_res"] = _participation_ratio(z_res).item()
                    log_stats["p_rank_utt"] = _participation_ratio(p_utt).item()
                row = {"step": step, **log_stats}

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
