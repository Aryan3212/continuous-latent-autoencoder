from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from data.augment import FeatureMaskConfig, MixConfig, WaveAugConfig, apply_feature_mask, apply_waveform_augment, maybe_mix_pair
from data.dataset import WebDatasetConfig, get_audio_wds, collate_fixed
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


def _lejepa_loss(center: torch.Tensor, view: torch.Tensor, return_per_sample: bool = False) -> torch.Tensor:
    # center, view: (B, D, T)
    diff = (center - view).pow(2).mean(dim=(1, 2))
    if return_per_sample:
        return diff
    return diff.mean()


def _pool(z: torch.Tensor) -> torch.Tensor:
    # z: (B,d,T') -> (B,2d)
    return torch.cat([z.mean(dim=-1), z.std(dim=-1, unbiased=False)], dim=1)


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _encode(model: torch.nn.ModuleDict, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    h0 = model["frontend"](wav)
    hE = model["encoder"](h0)
    # The latent z IS the encoder output hE.
    z = hE 
    return h0, hE, z, {} # No extra stats


def _decode(model: torch.nn.ModuleDict, z: torch.Tensor, target_len: int, sigma: torch.Tensor) -> torch.Tensor:
    z_dec = z
    if sigma.item() > 0:
        z_dec = z + torch.randn_like(z) * sigma
    return model["decoder"](z_dec, target_len=target_len)


def _primary_logits(e_mix: torch.Tensor, e_a: torch.Tensor, e_b: torch.Tensor) -> torch.Tensor:
    # cosine sim logits between e_mix and (e_a, e_b)
    em = F.normalize(e_mix, dim=-1)
    ea = F.normalize(e_a, dim=-1)
    eb = F.normalize(e_b, dim=-1)
    s_a = (em * ea).sum(dim=-1, keepdim=True)
    s_b = (em * eb).sum(dim=-1, keepdim=True)
    return torch.cat([s_a, s_b], dim=-1)  # (B,2)


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

    seed_all(int(cfg["run"]["seed"]))
    device = _select_device(cfg)

    run_id = cfg["run"].get("run_id") or _now_run_id()
    out_root = pathlib.Path(cfg["run"]["out_dir"]) / run_id
    ckpt_dir = out_root / "checkpoints"
    log_dir = out_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jsonl = JsonlLogger(str(log_dir / "train.jsonl"))
    wb = maybe_init_wandb(cfg, run_id, str(out_root))
    
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
    sigreg_cfg = cfg["loss"]["sigreg"].copy()
    if "weight" in sigreg_cfg:
        del sigreg_cfg["weight"]
    sigreg = SIGReg(latent_dim, SIGRegConfig(**sigreg_cfg))

    model = torch.nn.ModuleDict(
        {
            "frontend": frontend,
            "encoder": encoder,
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
    sig_w = float(cfg["loss"]["sigreg"]["weight"])
    stft_w = float(cfg["loss"].get("stft_weight", 1.0))
    wav_l1_w = float(cfg["loss"].get("wav_l1_weight", 0.0))

    use_lejepa_on_hE = bool(cfg["loss"].get("use_lejepa_on_hE", False))
    kl_weight = float(cfg["loss"].get("kl_weight", 0.0))

    mix_recon_cfg = cfg["loss"].get("mix_recon") or {}
    mix_recon_enabled = bool(mix_recon_cfg.get("enabled", False))
    mix_recon_w = float(mix_recon_cfg.get("weight", 1.0))
    mix_recon_start = int(mix_recon_cfg.get("start_step", 0))

    primary_cfg = cfg["loss"].get("primary") or {}
    primary_enabled = bool(primary_cfg.get("enabled", False))
    primary_w = float(primary_cfg.get("weight", 0.0))

    mix_view_w = float(jcfg.get("mix_view_weight", 1.0))
    gan_start = int(gan_cfg.get("start_step", 0))
    g_adv_w_max = float(gan_cfg.get("g_adv_weight", 1.0))
    fm_w_max = float(gan_cfg.get("fm_weight", 2.0))
    gan_warmup_steps = int(gan_cfg.get("warmup_steps", 5000))

    best: Dict[str, float] = {"val_jepa": float("inf"), "asr_wer": float("inf"), "composite": -float("inf")}
    if isinstance(resume_best, dict):
        for k in ["val_jepa", "asr_wer", "composite"]:
            if k in resume_best:
                best[k] = float(resume_best[k])

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
        sums = {"val_stft": 0.0, "val_jepa": 0.0, "val_sig": 0.0}
        n = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for vb in val_dl:
                vw = vb["wav"].to(device)
                h0, hE, z, stats = _encode(model, vw)
                h0m = apply_feature_mask(h0, feat_mask_cfg)
                hEm = model["encoder"](h0m)
                zm = hEm
                v_jepa = _lejepa_loss(z, zm)
                v_sig_a, _ = sigreg(z, step=step)
                v_sig_m, _ = sigreg(zm, step=step)
                v_sig = 0.5 * (v_sig_a + v_sig_m)
                xh = _decode(model, z, target_len=vw.size(-1), sigma=torch.tensor(0.0, device=device))
                v_stft, _ = stft(xh, vw)
                sums["val_stft"] += float(v_stft.detach().cpu())
                sums["val_jepa"] += float(v_jepa.detach().cpu())
                sums["val_sig"] += float(v_sig.detach().cpu())
                n += 1
                if val_batches is not None and n >= val_batches:
                    break
        model.train()
        return {k: v / max(1, n) for k, v in sums.items()}

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
        mix_it = iter(train_dl)
        
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
                    batch_b = next(mix_it)
                except StopIteration:
                    mix_it = iter(train_dl)
                    batch_b = next(mix_it)
                
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
                    # Clean view (V0) - NO augmentation on target, but maybe on input if we want denoising?
                    # The user asked for "augmentations".
                    # Standard RAE/LeJEPA: "View" is augmented, "Center" is clean(er).
                    # Current code: _encode(model, wav_a) -> z_a.
                    # If we want Denoising, we should encode augmented, decode to clean.
                    # But LeJEPA logic in this file: z_mask matches z_a.
                    # If we augment wav_a, z_a changes.
                
                    # Let's apply augmentation to get wav_aug.
                    # wav_a is CLEAN target.
                    wav_aug = apply_waveform_augment(wav_a, int(dcfg["sample_rate"]), wave_aug_cfg)
                
                    # Encode Clean (Center)
                    h0_a, hE_a, z_a, stats_a = _encode(model, wav_a)
                
                    # Encode Augmented (View 1)
                    # If we just want simple denoising, we can use wav_aug as input for "clean view" path
                    # but "clean view" is used as target for reconstruction in current code: 
                    # x_hat = _decode(z_a) -> loss(x_hat, wav_a)
                
                    # To implement "Denoising", we should encode wav_aug -> z_aug -> decode -> wav_a
                    # But we also have LeJEPA z_aug vs z_clean.
                
                    # Let's define:
                    # Center = Clean
                    # View = Augmented + Masked
                
                    h0_aug = model["frontend"](wav_aug)

                    # Masked feature view (V1) - applied to Augmented view features
                    h0_masked = apply_feature_mask(h0_aug, feat_mask_cfg)
                    hE_mask = model["encoder"](h0_masked)
                    z_mask = hE_mask

                    # LeJEPA: masked+augmented view should match clean center
                    if use_lejepa_on_hE:
                        l_jepa_mask_ps = _lejepa_loss(hE_a, hE_mask, return_per_sample=True)
                    else:
                        l_jepa_mask_ps = _lejepa_loss(z_a, z_mask, return_per_sample=True)
                    l_jepa_mask = l_jepa_mask_ps.mean()
                
                    # KL Divergence (on CLEAN or AUG? Usually on the one used for gen/prior. Clean makes sense for "prior" matching)
                    l_kl = torch.tensor(0.0, device=device)
                    if "mu" in stats_a and "logvar" in stats_a and float(cfg["loss"].get("kl_weight", 0.0)) > 0:
                        mu, logvar = stats_a["mu"], stats_a["logvar"]
                        l_kl_ps = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(dim=-1)
                        l_kl = l_kl_ps.mean()

                    l_jepa_mix = torch.tensor(0.0, device=device)
                    l_primary = torch.tensor(0.0, device=device)
                    l_stft_mix = torch.tensor(0.0, device=device)

                    # Exp1+: mix view (V2)
                    if mix_cfg.enabled and has_mix:
                        _, _, z_b, _ = _encode(model, wav_b)
                        _, _, z_mix, _ = _encode(model, wav_mix)

                        pidx_view = primary_idx.view(-1, *([1]*(z_a.ndim - 1)))
                        z_tgt_mix = torch.where(pidx_view == 1, z_b, z_a)

                        mix_mask_f = mixed_mask.float()
                        mix_mask_sum = mix_mask_f.sum().clamp(min=1e-8)

                        l_jepa_mix_ps = _lejepa_loss(z_tgt_mix, z_mix, return_per_sample=True)
                        l_jepa_mix = (l_jepa_mix_ps * mix_mask_f).sum() / mix_mask_sum

                        if mix_recon_enabled and step >= mix_recon_start:
                            sigma_mix = _latent_noise_sigma(cfg, step, device)
                            x_hat_mix = _decode(
                                model, z_mix, target_len=wav_tgt.size(-1), sigma=sigma_mix
                            )
                            l_stft_mix_ps, _ = stft(x_hat_mix, wav_tgt, return_per_sample=True)
                            l_stft_mix = (l_stft_mix_ps * mix_mask_f).sum() / mix_mask_sum

                        if primary_enabled:
                            e_mix = _pool(z_mix)
                            e_a = _pool(z_a)
                            e_b = _pool(z_b)
                            logits = _primary_logits(e_mix, e_a, e_b)
                            l_primary_ps = F.cross_entropy(logits, primary_idx, reduction='none')
                            l_primary = (l_primary_ps * mix_mask_f).sum() / mix_mask_sum

                    # SIGReg on clean embeddings
                    sig_losses = []
                    l_sig_a, sig_stats_a = sigreg(z_a, step=step)
                    sig_losses.append(l_sig_a)
                    # SIGReg on masked/augmented embeddings? Usually just one view is enough or both.
                    # Let's keep it on both to ensure isotropy everywhere.
                    l_sig_m, sig_stats_m = sigreg(z_mask, step=step)
                    sig_losses.append(l_sig_m)
                    if mix_cfg.enabled and has_mix:
                        l_sig_mix, _ = sigreg(z_mix, step=step)
                        sig_losses.append(l_sig_mix)
                    l_sig = torch.stack(sig_losses).mean()
                    sig_stats = {
                        "sigreg_clean": sig_stats_a["sigreg_loss"],
                        "sigreg_masked": sig_stats_m["sigreg_loss"],
                        "z_var_min": sig_stats_a["z_var_min"],
                        "z_var_med": sig_stats_a["z_var_med"],
                        "z_var_max": sig_stats_a["z_var_max"],
                        "z_var_penalty": sig_stats_a["z_var_penalty"],
                    }

                    # Decoder
                    sigma = _latent_noise_sigma(cfg, step, device)
                    x_hat = _decode(model, z_a, target_len=wav_a.size(-1), sigma=sigma)
                
                    t_stfts = batch.get("target_stfts")
                    l_stft_ps, stft_stats_ps = stft(x_hat, wav_a, return_per_sample=True, target_mags=t_stfts)
                    l_stft = l_stft_ps.mean()
                    stft_stats = {k: v.mean().detach() for k, v in stft_stats_ps.items()}

                    l_wav_ps = (x_hat - wav_a).abs().mean(dim=(1, 2))
                    l_wav = l_wav_ps.mean()

                    l_g_adv = torch.tensor(0.0, device=device)
                    l_fm = torch.tensor(0.0, device=device)
                    l_d = torch.tensor(0.0, device=device)
                    g_adv_w = 0.0
                    fm_w = 0.0
                    if gan_enabled and discriminators is not None and step >= gan_start:
                        gan_progress = min(1.0, (step - gan_start) / max(1, gan_warmup_steps))
                        g_adv_w = g_adv_w_max * gan_progress
                        fm_w = fm_w_max * gan_progress
                        _set_requires_grad(discriminators, True)
                        d_real_mpd, fmap_real_mpd = discriminators["mpd"](wav_a)
                        d_fake_mpd, fmap_fake_mpd = discriminators["mpd"](x_hat.detach())
                        d_real_msd, fmap_real_msd = discriminators["msd"](wav_a)
                        d_fake_msd, fmap_fake_msd = discriminators["msd"](x_hat.detach())
                        l_d = discriminator_loss(d_real_mpd, d_fake_mpd) + discriminator_loss(
                            d_real_msd, d_fake_msd
                        )
                        d_scaler.scale(l_d / grad_accum).backward()
                        _set_requires_grad(discriminators, False)
                        d_fake_mpd_g, fmap_fake_mpd_g = discriminators["mpd"](x_hat)
                        d_fake_msd_g, fmap_fake_msd_g = discriminators["msd"](x_hat)
                        fmap_real_mpd_det = [[f.detach() for f in layer] for layer in fmap_real_mpd]
                        fmap_real_msd_det = [[f.detach() for f in layer] for layer in fmap_real_msd]
                        l_g_adv = generator_loss(d_fake_mpd_g) + generator_loss(d_fake_msd_g)
                        l_fm = feature_matching_loss(fmap_real_mpd_det, fmap_fake_mpd_g) + feature_matching_loss(
                            fmap_real_msd_det, fmap_fake_msd_g
                        )

                    l_jepa = l_jepa_mask + mix_view_w * l_jepa_mix

                    loss = (
                        stft_w * l_stft
                        + wav_l1_w * l_wav
                        + jepa_w * l_jepa
                        + sig_w * l_sig
                        + g_adv_w * l_g_adv
                        + fm_w * l_fm
                        + kl_weight * l_kl
                        + (mix_recon_w * l_stft_mix if (mix_recon_enabled and step >= mix_recon_start) else 0.0)
                        + (primary_w * l_primary if primary_enabled else 0.0)
                    )
                    loss = loss / grad_accum

                g_scaler.scale(loss).backward()
                total_loss = total_loss + loss.detach()

                mb_step_stats = {
                    "l_stft": l_stft.detach(),
                    "l_wav": l_wav.detach(),
                    "l_jepa": l_jepa.detach(),
                    "l_jepa_mask": l_jepa_mask.detach(),
                    "l_sig": l_sig.detach(),
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
                        "g_adv_w": float(g_adv_w),
                        "fm_w": float(fm_w),
                    })

                if kl_weight > 0:
                    mb_step_stats["l_kl"] = l_kl.detach()

                unique_ds = set(dataset_names)
                if len(unique_ds) > 1:
                    for ds_name in unique_ds:
                        indices = [i for i, n in enumerate(dataset_names) if n == ds_name]
                        if not indices:
                            continue
                        idx_t = torch.tensor(indices, device=device)
                        mb_step_stats[f"loss_stft/{ds_name}"] = l_stft_ps[idx_t].mean().detach()
                        mb_step_stats[f"loss_wav/{ds_name}"] = l_wav_ps[idx_t].mean().detach()
                        mb_step_stats[f"loss_jepa/{ds_name}"] = l_jepa_mask_ps[idx_t].mean().detach()

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

            log_interval = int(cfg["train"]["log_interval_steps"])
            if step % log_interval == 0:
                log_stats = {}
                for k, v in accum_stats.items():
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(v, op=dist.ReduceOp.AVG)
                    log_stats[k] = v.item() / log_interval
                    v.zero_()
                row = {"step": step, **log_stats}
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
