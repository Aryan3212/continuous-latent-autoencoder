from __future__ import annotations

import argparse
import math
import os
import pathlib
import signal
import sys
import time
from typing import Any, Dict, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import torch.nn.functional as F

from data.augment import (
    WaveAugConfig,
    WaveChunkMaskConfig,
    apply_waveform_augment,
    apply_waveform_chunk_mask,
    make_frame_chunk_masks,
)
from data.dataset import AudioDataset, DatasetConfig, collate_fixed
from losses.multires_stft import MultiResSTFTConfig, MultiResSTFTLoss
from models.decoder_generator import DecoderConfig, WaveformDecoder
from models.encoder import Encoder, EncoderConfig
from models.frontend_conv import ConvFrontend, FrontendConfig
from models.mhc import sinkhorn_log
from models.projector import Projector, ProjectorConfig
from models.sigreg import SIGReg, SIGRegConfig
from utils.checkpoint import save_checkpoint, save_run_metadata, try_git_hash
from utils.config import apply_overrides, load_config
from utils.logging import JsonlLogger, maybe_init_wandb
from utils.schema import Config
from utils.seed import seed_all


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _select_device(cfg: Config) -> torch.device:
    want = cfg.run.device
    if want in ("cpu", "cuda"):
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _global_local_jepa_loss(
    p_cat: torch.Tensor,
    local_mask_cat: torch.Tensor,
    num_globals: int,
    num_locals: int,
    lam_context: float = 1.0,
    distance_weight: bool = True,
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
    if G < 1 or L < 1:
        raise ValueError(f"need >=1 global and >=1 local view, got G={G}, L={L}")
    V = G + L
    VB, D, T = p_cat.shape
    B = VB // V

    p = p_cat.view(V, B, D, T)
    p_g = p[:G]                                     # (G, B, D, T)
    p_l = p[G:]                                     # (L, B, D, T)

    center = p_g.mean(dim=0, keepdim=True)          # (1, B, D, T)  globals-only anchor

    # Cross-global consistency: pull each global to the anchor on every frame.
    err_g = (p_g - center).pow(2).mean(dim=2)       # (G, B, T)
    l_global = err_g.mean()

    # Locals: predict + context against the globals-only center.
    err_l = (p_l - center).pow(2).mean(dim=2)       # (L, B, T)
    m = local_mask_cat.view(L, B, T).to(p.dtype)    # 1 on masked frames
    one_minus_m = 1.0 - m

    pred_num = (err_l * m).sum()
    pred_den = m.sum().clamp_min(1.0)
    l_predict = pred_num / pred_den

    if distance_weight:
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
    else:
        ctx_w = one_minus_m

    ctx_num = (err_l * ctx_w).sum()
    ctx_den = ctx_w.sum().clamp_min(1e-6)
    l_context = ctx_num / ctx_den

    loss = l_global + l_predict + lam_context * l_context
    return loss, l_global.detach(), l_predict.detach(), l_context.detach()


def _decode(model: torch.nn.ModuleDict, z: torch.Tensor, target_len: int) -> torch.Tensor:
    return model["decoder"](z, target_len=target_len)


def main() -> None:
    # Flush stdout immediately so print() output appears over SSH/nohup/pipe.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    _shutdown = False

    def _sigint_handler(sig: int, frame: object) -> None:
        nonlocal _shutdown
        print("\n[train] SIGINT received — stopping after current step.", flush=True)
        _shutdown = True

    signal.signal(signal.SIGINT, _sigint_handler)

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
    ap.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler with W&B")
    ap.add_argument("--profile_wait", type=int, default=0, help="Steps to wait before profiling")
    ap.add_argument("--profile_warmup", type=int, default=0, help="Steps to warm up profiler")
    ap.add_argument("--profile_active", type=int, default=1, help="Steps to actively profile")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    cfg.resolved_config_path = args.config

    # Resume path layout: <out_dir>/<run_id>/checkpoints/<name>.pt
    # Infer the existing run_id so we reuse its out_dir and wandb run.
    if args.resume:
        resume_path = pathlib.Path(args.resume)
        if resume_path.parent.name == "checkpoints":
            inferred_run_id = resume_path.parent.parent.name
            if not cfg.run.run_id:
                cfg.run.run_id = inferred_run_id

    seed_all(cfg.run.seed)
    device = _select_device(cfg)

    run_id = cfg.run.run_id or _now_run_id()
    out_root = pathlib.Path(cfg.run.out_dir) / run_id
    ckpt_dir = out_root / "checkpoints"
    log_dir = out_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jsonl = JsonlLogger(str(log_dir / "train.jsonl"))
    wb = maybe_init_wandb(cfg, run_id, str(out_root), resume=bool(args.resume))

    # Data
    dcfg = cfg.data
    if dcfg.train_manifest is None:
        raise ValueError("Set data.train_manifest to a JSONL manifest path (e.g. data/manifests/train.jsonl)")
    meta_extra = {
        "git_hash": try_git_hash(cwd=str(pathlib.Path(".").resolve())),
        "train_manifest": str(dcfg.train_manifest),
        "val_manifest": str(dcfg.val_manifest or ""),
    }
    save_run_metadata(str(out_root), cfg, extra=meta_extra)
    train_ds = AudioDataset(
        DatasetConfig(
            manifest=dcfg.train_manifest,
            sample_rate=dcfg.sample_rate,
            segment_seconds=dcfg.segment_seconds,
            random_crop=True,
        )
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        num_workers=dcfg.num_workers,
        pin_memory=dcfg.pin_memory,
        persistent_workers=dcfg.persistent_workers if dcfg.num_workers > 0 else False,
        # "spawn" avoids inheriting CUDA's locked mutexes from the parent process.
        # fork-after-CUDA-init deadlocks worker processes on Linux (the default).
        multiprocessing_context="spawn" if dcfg.num_workers > 0 else None,
        collate_fn=collate_fixed,
        drop_last=True,
        shuffle=True,
    )

    # Model
    mcfg = cfg.model
    frontend = ConvFrontend(FrontendConfig(**mcfg.frontend.model_dump()))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg.encoder.model_dump()))

    latent_dim = mcfg.encoder.d_model

    decoder = WaveformDecoder(latent_dim, DecoderConfig(**mcfg.decoder.model_dump()))

    projector = Projector(latent_dim, ProjectorConfig(**mcfg.projector.model_dump()))
    proj_dim = projector.output_dim

    sreg = cfg.loss.sigreg
    sigreg = SIGReg(proj_dim, SIGRegConfig(num_slices=sreg.num_slices, t_max=sreg.t_max, n_points=sreg.n_points))

    model = torch.nn.ModuleDict(
        {
            "frontend": frontend,
            "encoder": encoder,
            "projector": projector,
            "decoder": decoder,
            "sigreg": sigreg,
        }
    ).to(device)

    # Losses
    stft = MultiResSTFTLoss(MultiResSTFTConfig(**cfg.loss.stft.model_dump())).to(device)
    wave_aug_cfg = WaveAugConfig(**cfg.aug.wave_aug.model_dump())
    wave_chunk_mask_cfg = WaveChunkMaskConfig(**cfg.aug.wave_chunk_mask.model_dump())

    # Frontend stride product gives samples-per-output-frame. Used to convert
    # frame-level chunk masks (local-view recipe) into waveform sample masks.
    samples_per_frame = math.prod(mcfg.frontend.strides)
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

    use_amp = cfg.run.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        if scheduler and (state.get("extra") or {}).get("scheduler"):
            scheduler.load_state_dict(state["extra"]["scheduler"])
        step = int(state.get("step", 0))

    # CLI overrides for loop intervals.
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.log_interval_steps is not None:
        cfg.train.log_interval_steps = args.log_interval_steps
    if args.eval_interval_steps is not None:
        cfg.train.eval_interval_steps = args.eval_interval_steps
    if args.save_interval_steps is not None:
        cfg.train.save_interval_steps = args.save_interval_steps

    # Training loop
    model.train()
    max_steps = cfg.train.max_steps
    grad_accum = cfg.train.grad_accum_steps
    grad_clip = cfg.optim.grad_clip
    val_batches = cfg.train.val_batches

    jcfg = cfg.loss.jepa
    jepa_w = jcfg.weight
    num_globals = jcfg.num_globals
    num_locals = jcfg.num_locals
    if num_globals < 1 or num_locals < 1:
        raise ValueError(f"loss.jepa.num_globals and num_locals must both be >= 1; got G={num_globals}, L={num_locals}")
    sig_w = cfg.loss.sigreg.weight
    sig_z_w = cfg.loss.sigreg.z_weight
    sig_utt_w = cfg.loss.sigreg.utt_weight
    stft_w = cfg.loss.stft_weight
    wav_l1_w = cfg.loss.wav_l1_weight

    # V-JEPA 2.1 context-loss weight: relative weight of L_context vs L_predict
    # in the Dense Predictive Loss. Paper uses ~1.0 with distance weighting.
    lam_context_w = jcfg.context_weight

    def _validate_one() -> Dict[str, float]:
        if not dcfg.val_manifest:
            return {}
        val_ds = AudioDataset(
            DatasetConfig(
                manifest=dcfg.val_manifest,
                sample_rate=dcfg.sample_rate,
                segment_seconds=dcfg.segment_seconds,
                random_crop=False,
            )
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            num_workers=0,
            collate_fn=collate_fixed,
            drop_last=True,
        )
        model.eval()
        sums = {
            "val_stft": torch.tensor(0.0, device=device),
            "val_sig": torch.tensor(0.0, device=device),
        }
        n = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for vb in val_dl:
                vw = vb["wav"].to(device)
                z = model["encoder"](model["frontend"](vw))
                p_clean = model["projector"](z)
                def _flatten(t: torch.Tensor) -> torch.Tensor:
                    return t.permute(0, 2, 1).reshape(-1, t.size(1))
                fp = _flatten(p_clean)
                v_sig_f, _ = sigreg(fp, step=step); v_sig_f = v_sig_f / max(1, fp.size(0))
                xh = _decode(model, z, target_len=vw.size(-1))
                v_stft, _ = stft(xh, vw)
                sums["val_stft"] += v_stft.detach()
                sums["val_sig"] += v_sig_f.detach()
                n += 1
                if val_batches is not None and n >= val_batches:
                    break
        model.train()
        return {k: (v / max(1, n)).item() for k, v in sums.items()}

    def _extra_state(**extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"scheduler": scheduler.state_dict() if scheduler else None}
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

    train_it = iter(train_dl)
    while step < max_steps and not _shutdown:
        optimizer.zero_grad(set_to_none=True)

        total_loss = torch.tensor(0.0, device=device)
        mb_stats: Dict[str, Any] = {}

        microbatches = []
        for _ in range(grad_accum):
            try:
                batch = next(train_it)
            except StopIteration:
                train_it = iter(train_dl)
                batch = next(train_it)
            microbatches.append(batch)

        for batch in microbatches:
            wav_a = batch["wav"]  # (B,1,T)
            dataset_names = [m.get("dataset", "unknown") for m in batch["meta"]]

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
                # Locals already carry waveform masking; globals must stay clean
                # so the globals-only center remains the anchor.
                z_cat = model["encoder"](h0_cat)                      # (V*B, D, T')
                p_cat = model["projector"](z_cat)                     # (V*B, P, T')

                # Align frame-mask to actual encoder grid (T'). It was generated
                # at n_frames_per_segment; conv arithmetic may differ by ±1.
                T_actual = p_cat.size(-1)
                if local_frame_masks.size(-1) != T_actual:
                    local_frame_masks = F.interpolate(
                        local_frame_masks.unsqueeze(1),
                        size=T_actual,
                        mode="nearest",
                    ).squeeze(1)
                local_mask_cat = local_frame_masks.unsqueeze(1)        # (L*B, 1, T')

                l_jepa_mask, l_jepa_global_dbg, l_jepa_pred_dbg, l_jepa_ctx_dbg = _global_local_jepa_loss(
                    p_cat,
                    local_mask_cat,
                    num_globals=num_globals,
                    num_locals=num_locals,
                    lam_context=lam_context_w,
                    distance_weight=True,
                )

                # ---- SIGReg (LeWM-style, on projector out + encoder z) ---------
                # Frame-level on p: reshape p_cat (V*B, P, T') -> (T'*V*B, P) so
                # SIGReg treats each (frame, view, sample) triple as an
                # independent point. Paper N-scaling is left to SIGReg itself;
                # no extra /N rescaling here.
                D_lat = p_cat.size(1)
                sig_input = p_cat.permute(2, 0, 1)                # (T', V*B, P)
                l_sig, _ = sigreg(
                    sig_input.reshape(-1, D_lat),                 # (T'*V*B, P)
                    step=step,
                )
                # Frame-level on z: the projector bottleneck hides encoder dims
                # from every post-projector loss, so anti-collapse pressure must
                # also act on z directly. SlicingUnivariateTest sizes its slice
                # matrix from x.size(-1), so the shared instance handles D != P.
                if sig_z_w > 0:
                    z_sig_input = z_cat.permute(2, 0, 1)          # (T', V*B, D)
                    l_sig_z, _ = sigreg(
                        z_sig_input.reshape(-1, z_cat.size(1)),   # (T'*V*B, D)
                        step=step,
                    )
                else:
                    l_sig_z = torch.tensor(0.0, device=device)
                # Utterance-level on z: mean-pool over T' so pooled embeddings
                # (gender/emotion/accent probes) can't collapse to a single
                # point while per-frame stats stay healthy. Must act on z, not
                # p — the probes pool z, and the projector can satisfy a
                # p-level term without pooled z gaining any variance.
                if sig_utt_w > 0:
                    l_sig_utt, _ = sigreg(
                        z_cat.mean(dim=2),                        # (V*B, D)
                        step=step,
                    )
                else:
                    l_sig_utt = torch.tensor(0.0, device=device)
                sig_stats = {
                    "l_sig_frm": l_sig.detach(),
                    "l_sig_z": l_sig_z.detach(),
                    "l_sig_utt": l_sig_utt.detach(),
                    "l_jepa_predict": l_jepa_pred_dbg,
                    "l_jepa_context": l_jepa_ctx_dbg,
                    "l_jepa_global": l_jepa_global_dbg,
                }

                # Diagnostic slicing: compare global-0 vs local-0 (clean vs masked signal).
                z_a = z_cat[:B]               # view-0 encoder embeddings (decoder + rank diag)
                p_a = p_cat[:B]               # view-0 projected (JEPA-space diagnostics)
                mask_idx = num_globals
                p_mask = p_cat[mask_idx * B : (mask_idx + 1) * B]

                # Decode from view-0 to clean wav_a (denoising reconstruction).
                x_hat = _decode(model, z_a, target_len=wav_a.size(-1))

                l_stft_ps, stft_stats_ps = stft(x_hat, wav_a, return_per_sample=True)
                l_stft = l_stft_ps.mean()
                stft_stats = {k: v.mean().detach() for k, v in stft_stats_ps.items()}

                l_wav_ps = (x_hat - wav_a).abs().mean(dim=(1, 2))
                l_wav = l_wav_ps.mean()

            with torch.amp.autocast("cuda", enabled=use_amp):
                l_jepa = l_jepa_mask

                loss = (
                    stft_w * l_stft
                    + wav_l1_w * l_wav
                    + jepa_w * l_jepa
                    + sig_w * l_sig
                    + sig_z_w * l_sig_z
                    + sig_utt_w * l_sig_utt
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            total_loss = total_loss + loss.detach()

            # Diagnostic metrics for collapse detection.
            # Eigendecompositions are expensive — only compute on log boundaries.
            log_interval = cfg.train.log_interval_steps
            compute_ranks = (step % log_interval == 0)
            with torch.no_grad():
                if compute_ranks:
                    z_flat = z_a.permute(0, 2, 1).reshape(-1, z_a.size(1))
                    z_centered = z_flat - z_flat.mean(dim=0)
                    z_cov = (z_centered.T @ z_centered) / (z_flat.size(0) - 1)
                    # clamp_min(0): eigvalsh can emit tiny negative eigenvalues on
                    # near-singular covariances, which pushes the participation
                    # ratio below its true floor of 1.
                    z_eigvals = torch.linalg.eigvalsh(z_cov).clamp_min(0)
                    z_rank = (z_eigvals.sum()**2) / (z_eigvals.pow(2).sum() + 1e-8)

                    z_utt = z_a.mean(dim=2)
                    z_utt_c = z_utt - z_utt.mean(dim=0)
                    z_utt_cov = (z_utt_c.T @ z_utt_c) / max(z_utt.size(0) - 1, 1)
                    z_utt_eig = torch.linalg.eigvalsh(z_utt_cov).clamp_min(0)
                    z_rank_utt = (z_utt_eig.sum()**2) / (z_utt_eig.pow(2).sum() + 1e-8)

                    z_res = z_a - z_a.mean(dim=2, keepdim=True)
                    z_res_flat = z_res.permute(0, 2, 1).reshape(-1, z_a.size(1))
                    z_res_cov = (z_res_flat.T @ z_res_flat) / max(z_res_flat.size(0) - 1, 1)
                    z_res_eig = torch.linalg.eigvalsh(z_res_cov).clamp_min(0)
                    z_rank_res = (z_res_eig.sum()**2) / (z_res_eig.pow(2).sum() + 1e-8)
                else:
                    z_rank = torch.tensor(0.0, device=z_a.device)
                    z_rank_utt = torch.tensor(0.0, device=z_a.device)
                    z_rank_res = torch.tensor(0.0, device=z_a.device)

                # JEPA collapse detector — cheap, run every microbatch. Raw z/p
                # RMS values are pinned ~1 by LayerNorm/BatchNorm and carry no
                # signal; the informative quantity is diff relative to norm.
                p_a_rms = p_a.pow(2).mean().sqrt()
                jepa_diff_rms = (p_a - p_mask).pow(2).mean().sqrt()
                jepa_to_norm_ratio = jepa_diff_rms / p_a_rms.clamp_min(1e-6)

            mb_step_stats = {
                "l_stft": l_stft.detach(),
                "l_wav": l_wav.detach(),
                "l_jepa": l_jepa.detach(),
                "z_rank": z_rank.detach(),
                "z_rank_utt": z_rank_utt.detach(),
                "z_rank_res": z_rank_res.detach(),
                "jepa_diff_rms": jepa_diff_rms.detach(),
                "jepa_to_norm_ratio": jepa_to_norm_ratio.detach(),
                "vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            }
            mb_step_stats.update({k: v.detach() for k, v in stft_stats.items()})
            mb_step_stats.update({k: v.detach() for k, v in sig_stats.items()})

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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for k, v in stats.items():
            if k not in accum_stats:
                accum_stats[k] = torch.tensor(0.0, device=device)
            if isinstance(v, torch.Tensor):
                accum_stats[k] += v.detach()
            else:
                accum_stats[k] += torch.tensor(v, device=device)

        step += 1
        if scheduler is not None:
            scheduler.step()

        log_interval = cfg.train.log_interval_steps
        if step % log_interval == 0:
            log_stats = {}
            for k, v in accum_stats.items():
                log_stats[k] = v.item() / log_interval
                v.zero_()
            row = {"step": step, **log_stats}

            encoder_mod = model["encoder"]
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

        if step % cfg.train.save_interval_steps == 0:
            save_checkpoint(
                str(ckpt_dir / "last.pt"),
                step=step,
                model=model,
                optimizer=optimizer,
                scaler=scaler if scaler.is_enabled() else None,
                cfg=cfg,
                extra=_extra_state(),
            )

        if step % cfg.train.eval_interval_steps == 0:
            v = _validate_one()
            if v:
                row = {"step": step, **v}
                jsonl.log(row)
                if wb is not None:
                    wb.log(row, step=step)

        if prof is not None:
            prof.step()

    if prof is not None:
        prof.stop()

    save_checkpoint(
        str(ckpt_dir / "last.pt"),
        step=step,
        model=model,
        optimizer=optimizer,
        scaler=scaler if scaler.is_enabled() else None,
        cfg=cfg,
        extra=_extra_state(),
    )


if __name__ == "__main__":
    main()
