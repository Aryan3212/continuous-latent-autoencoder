from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from data.augment import FeatureMaskConfig, MixConfig, apply_feature_mask, maybe_mix_pair
from data.dataset import AudioManifestDataset, ManifestConfig, collate_fixed
from losses.multires_stft import MultiResSTFTConfig, MultiResSTFTLoss
from models.decoder_generator import DecoderConfig, WaveformDecoder
from models.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)
from models.encoder import Bottleneck, Encoder, EncoderConfig
from models.frontend_conv import ConvFrontend, FrontendConfig
from models.sigreg import SIGReg, SIGRegConfig
from optim.lr_schedulers import Eden, Eden2
from optim.scaled_adam import ScaledAdam
from utils.checkpoint import save_checkpoint, save_run_metadata, sha256_file, try_git_hash
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


def _lejepa_loss(center: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    return (center - view).pow(2).mean()


def _pool(z: torch.Tensor) -> torch.Tensor:
    # z: (B,d,T') -> (B,2d)
    return torch.cat([z.mean(dim=-1), z.std(dim=-1, unbiased=False)], dim=1)


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _encode(model: torch.nn.ModuleDict, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h0 = model["frontend"](wav)
    hE = model["encoder"](h0)
    z = model["bottleneck"](hE)
    return h0, hE, z


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--log_interval_steps", type=int, default=None)
    ap.add_argument("--eval_interval_steps", type=int, default=None)
    ap.add_argument("--save_interval_steps", type=int, default=None)
    ap.add_argument("--run_eval_on_save", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

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

    # Data
    dcfg = cfg["data"]
    if dcfg.get("train_manifest") is None:
        raise ValueError("Set data.train_manifest=/path/train.jsonl")
    meta_extra = {
        "git_hash": try_git_hash(cwd=str(pathlib.Path(".").resolve())),
        "train_manifest": str(dcfg["train_manifest"]),
        "train_manifest_sha256": sha256_file(dcfg["train_manifest"]),
        "val_manifest": str(dcfg.get("val_manifest") or ""),
        "val_manifest_sha256": sha256_file(dcfg["val_manifest"]) if dcfg.get("val_manifest") else "",
    }
    save_run_metadata(str(out_root), cfg, extra=meta_extra)
    train_ds = AudioManifestDataset(
        ManifestConfig(
            manifest_path=dcfg["train_manifest"],
            sample_rate=int(dcfg["sample_rate"]),
            segment_seconds=float(dcfg["segment_seconds"]),
        )
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=bool(dcfg.get("pin_memory", True)),
        collate_fn=collate_fixed,
        drop_last=True,
    )
    train_it = iter(train_dl)

    # Optional second iterator for mixing (pulls different samples cheaply).
    mix_cfg = MixConfig(**(cfg.get("aug", {}).get("mix", {}) or {}))
    mix_it = iter(train_dl)

    # Model
    mcfg = cfg["model"]
    frontend = ConvFrontend(FrontendConfig(**mcfg["frontend"]))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg["encoder"]))
    bottleneck = Bottleneck(
        in_dim=mcfg["encoder"]["d_model"],
        latent_dim=int(mcfg["bottleneck"]["latent_dim"]),
        norm=str(mcfg["bottleneck"]["norm"]),
    )
    decoder_cfg = DecoderConfig(**mcfg["decoder"])
    decoder = WaveformDecoder(int(mcfg["bottleneck"]["latent_dim"]), decoder_cfg)
    if decoder_cfg.latent_stats_path:
        stats = torch.load(decoder_cfg.latent_stats_path, map_location="cpu")
        decoder.set_latent_stats(stats["mean"], stats["var"])
    sigreg_cfg = cfg["loss"]["sigreg"].copy()
    if "weight" in sigreg_cfg:
        del sigreg_cfg["weight"]
    sigreg = SIGReg(int(mcfg["bottleneck"]["latent_dim"]), SIGRegConfig(**sigreg_cfg))

    model = torch.nn.ModuleDict(
        {
            "frontend": frontend,
            "encoder": encoder,
            "bottleneck": bottleneck,
            "decoder": decoder,
            "sigreg": sigreg,
        }
    ).to(device)

    gan_cfg = cfg.get("gan") or {}
    gan_enabled = bool(gan_cfg.get("enabled", False))
    discriminators = None
    d_optimizer = None
    if gan_enabled:
        mpd = MultiPeriodDiscriminator(periods=gan_cfg.get("periods", [2, 3, 5, 7, 11]))
        msd = MultiScaleDiscriminator(scales=int(gan_cfg.get("msd_scales", 3)))
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
        if state.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
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
    val_batches = int(cfg["train"].get("val_batches", 8))

    jcfg = cfg["loss"]["jepa"]
    jepa_w = float(jcfg["weight"])
    sig_w = float(cfg["loss"]["sigreg"]["weight"])
    wav_l1_w = float(cfg["loss"].get("wav_l1_weight", 0.0))

    mix_recon_cfg = cfg["loss"].get("mix_recon") or {}
    mix_recon_enabled = bool(mix_recon_cfg.get("enabled", False))
    mix_recon_w = float(mix_recon_cfg.get("weight", 1.0))
    mix_recon_start = int(mix_recon_cfg.get("start_step", 0))

    primary_cfg = cfg["loss"].get("primary") or {}
    primary_enabled = bool(primary_cfg.get("enabled", False))
    primary_w = float(primary_cfg.get("weight", 0.0))

    mix_view_w = float(jcfg.get("mix_view_weight", 1.0))
    gan_start = int(gan_cfg.get("start_step", 0))
    g_adv_w = float(gan_cfg.get("g_adv_weight", 1.0))
    fm_w = float(gan_cfg.get("fm_weight", 10.0))

    best: Dict[str, float] = {"val_jepa": float("inf"), "asr_wer": float("inf"), "composite": -float("inf")}
    if isinstance(resume_best, dict):
        for k in ["val_jepa", "asr_wer", "composite"]:
            if k in resume_best:
                best[k] = float(resume_best[k])

    def _validate_one() -> Dict[str, float]:
        if not dcfg.get("val_manifest"):
            return {}
        val_ds = AudioManifestDataset(
            ManifestConfig(
                manifest_path=dcfg["val_manifest"],
                sample_rate=int(dcfg["sample_rate"]),
                segment_seconds=float(dcfg["segment_seconds"]),
            )
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fixed,
        )
        model.eval()
        sums = {"val_stft": 0.0, "val_jepa": 0.0, "val_sig": 0.0}
        n = 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for vb in val_dl:
                vw = vb["wav"].to(device)
                h0, _, z = _encode(model, vw)
                h0m = apply_feature_mask(h0, feat_mask_cfg)
                hEm = model["encoder"](h0m)
                zm = model["bottleneck"](hEm)
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
                if n >= val_batches:
                    break
        model.train()
        return {k: v / max(1, n) for k, v in sums.items()}

    def _extra_state(**extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"best": dict(best), "scheduler": scheduler.state_dict() if scheduler else None}
        if gan_enabled and discriminators is not None:
            payload["discriminators"] = discriminators.state_dict()
            payload["d_optimizer"] = d_optimizer.state_dict() if d_optimizer else None
        payload.update(extra)
        return payload

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        if gan_enabled and d_optimizer is not None:
            d_optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        stats: Dict[str, Any] = {}

        for micro in range(grad_accum):
            try:
                batch = next(train_it)
            except StopIteration:
                train_it = iter(train_dl)
                batch = next(train_it)

            wav_a = batch["wav"]  # (B,1,T)

            # Optional mix view (Exp1+)
            try:
                batch_b = next(mix_it)
            except StopIteration:
                mix_it = iter(train_dl)
                batch_b = next(mix_it)
            wav_b = batch_b["wav"]

            # Build mix waveform + per-sample primary target when enabled.
            mixed_mask = torch.zeros((wav_a.size(0),), dtype=torch.bool)
            primary_idx = torch.zeros((wav_a.size(0),), dtype=torch.long)
            snr_db_vals = torch.zeros((wav_a.size(0),), dtype=torch.float32)
            wav_mix = wav_a
            wav_tgt = wav_a
            if mix_cfg.enabled and mix_cfg.prob > 0.0:
                wav_mix_list = []
                wav_tgt_list = []
                for i in range(wav_a.size(0)):
                    y, did, sdb, pidx = maybe_mix_pair(wav_a[i, 0], wav_b[i, 0], mix_cfg)
                    mixed_mask[i] = did
                    primary_idx[i] = pidx
                    snr_db_vals[i] = float(sdb)
                    wav_mix_list.append(y)
                    wav_tgt_list.append(wav_a[i, 0] if pidx == 0 else wav_b[i, 0])
                wav_mix = torch.stack(wav_mix_list, dim=0).unsqueeze(1)
                wav_tgt = torch.stack(wav_tgt_list, dim=0).unsqueeze(1)

            wav_a = wav_a.to(device, non_blocking=True)
            wav_b = wav_b.to(device, non_blocking=True)
            wav_mix = wav_mix.to(device, non_blocking=True)
            wav_tgt = wav_tgt.to(device, non_blocking=True)
            mixed_mask = mixed_mask.to(device)
            primary_idx = primary_idx.to(device)
            snr_db_vals = snr_db_vals.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Clean view (V0)
                h0_a, _, z_a = _encode(model, wav_a)

                # Masked feature view (V1)
                h0_masked = apply_feature_mask(h0_a, feat_mask_cfg)
                hE_mask = model["encoder"](h0_masked)
                z_mask = model["bottleneck"](hE_mask)

                # LeJEPA: masked view should match clean center
                l_jepa_mask = _lejepa_loss(z_a, z_mask)

                l_jepa_mix = torch.tensor(0.0, device=device)
                l_primary = torch.tensor(0.0, device=device)
                l_stft_mix = torch.tensor(0.0, device=device)

                # Exp1+: mix view (V2)
                if mix_cfg.enabled and bool(mixed_mask.any().item()):
                    _, _, z_b = _encode(model, wav_b)
                    _, _, z_mix = _encode(model, wav_mix)

                    z_tgt_mix = z_a.clone()
                    z_tgt_mix[primary_idx == 1] = z_b[primary_idx == 1]
                    l_jepa_mix = _lejepa_loss(z_tgt_mix[mixed_mask], z_mix[mixed_mask])

                    if mix_recon_enabled and step >= mix_recon_start:
                        sigma_mix = _latent_noise_sigma(cfg, step, device)
                        x_hat_mix = _decode(
                            model, z_mix[mixed_mask], target_len=wav_tgt.size(-1), sigma=sigma_mix
                        )
                        l_stft_mix, _ = stft(x_hat_mix, wav_tgt[mixed_mask])

                    if primary_enabled:
                        e_mix = _pool(z_mix[mixed_mask])
                        e_a = _pool(z_a[mixed_mask])
                        e_b = _pool(z_b[mixed_mask])
                        logits = _primary_logits(e_mix, e_a, e_b)
                        l_primary = F.cross_entropy(logits, primary_idx[mixed_mask])

                # SIGReg on clean embeddings
                sig_losses = []
                l_sig_a, sig_stats_a = sigreg(z_a, step=step)
                sig_losses.append(l_sig_a)
                l_sig_m, sig_stats_m = sigreg(z_mask, step=step)
                sig_losses.append(l_sig_m)
                if mix_cfg.enabled and bool(mixed_mask.any().item()):
                    l_sig_mix, _ = sigreg(z_mix[mixed_mask], step=step)
                    sig_losses.append(l_sig_mix)
                l_sig = torch.stack(sig_losses).mean()
                sig_stats = {
                    "sigreg_clean": sig_stats_a["sigreg_loss"],
                    "sigreg_masked": sig_stats_m["sigreg_loss"],
                }

                # Decoder (Exp0: reconstruct clean; Exp1+: optionally decode mixed->primary)
                sigma = _latent_noise_sigma(cfg, step, device)
                x_hat = _decode(model, z_a, target_len=wav_a.size(-1), sigma=sigma)
                l_stft, stft_stats = stft(x_hat, wav_a)

                l_wav = (x_hat - wav_a).abs().mean()

                l_g_adv = torch.tensor(0.0, device=device)
                l_fm = torch.tensor(0.0, device=device)
                l_d = torch.tensor(0.0, device=device)
                if gan_enabled and discriminators is not None and step >= gan_start:
                    _set_requires_grad(discriminators, True)
                    d_real_mpd, fmap_real_mpd = discriminators["mpd"](wav_a)
                    d_fake_mpd, fmap_fake_mpd = discriminators["mpd"](x_hat.detach())
                    d_real_msd, fmap_real_msd = discriminators["msd"](wav_a)
                    d_fake_msd, fmap_fake_msd = discriminators["msd"](x_hat.detach())
                    l_d = discriminator_loss(d_real_mpd, d_fake_mpd) + discriminator_loss(
                        d_real_msd, d_fake_msd
                    )
                    _set_requires_grad(discriminators, False)
                    d_fake_mpd_g, fmap_fake_mpd_g = discriminators["mpd"](x_hat)
                    d_fake_msd_g, fmap_fake_msd_g = discriminators["msd"](x_hat)
                    with torch.no_grad():
                        d_real_mpd_g, fmap_real_mpd_g = discriminators["mpd"](wav_a)
                        d_real_msd_g, fmap_real_msd_g = discriminators["msd"](wav_a)
                    l_g_adv = generator_loss(d_fake_mpd_g) + generator_loss(d_fake_msd_g)
                    l_fm = feature_matching_loss(fmap_real_mpd_g, fmap_fake_mpd_g) + feature_matching_loss(
                        fmap_real_msd_g, fmap_fake_msd_g
                    )

                l_jepa = l_jepa_mask + mix_view_w * l_jepa_mix

                loss = (
                    l_stft
                    + wav_l1_w * l_wav
                    + jepa_w * l_jepa
                    + sig_w * l_sig
                    + g_adv_w * l_g_adv
                    + fm_w * l_fm
                    + (mix_recon_w * l_stft_mix if (mix_recon_enabled and step >= mix_recon_start) else 0.0)
                    + (primary_w * l_primary if primary_enabled else 0.0)
                )
                loss = loss / grad_accum

            if gan_enabled and step >= gan_start:
                scaler.scale(l_d / grad_accum).backward()
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu())

            stats.update(
                {
                    "loss": total_loss,
                    "l_stft": float(l_stft.detach().cpu()),
                    "l_stft_mix": float(l_stft_mix.detach().cpu()),
                    "l_wav": float(l_wav.detach().cpu()),
                    "l_jepa": float(l_jepa.detach().cpu()),
                    "l_jepa_mask": float(l_jepa_mask.detach().cpu()),
                    "l_jepa_mix": float(l_jepa_mix.detach().cpu()),
                    "l_sig": float(l_sig.detach().cpu()),
                    "l_g_adv": float(l_g_adv.detach().cpu()),
                    "l_fm": float(l_fm.detach().cpu()),
                    "l_d": float(l_d.detach().cpu()),
                    "l_primary": float(l_primary.detach().cpu()),
                    "sigma": float(sigma.detach().cpu()),
                    "mixed_frac": float(mixed_mask.float().mean().detach().cpu()),
                    "snr_db_mean": float(snr_db_vals[mixed_mask].mean().detach().cpu())
                    if bool(mixed_mask.any().item())
                    else 0.0,
                    "z_mean": float(z_a.mean().detach().cpu()),
                    "z_std": float(z_a.std(unbiased=False).detach().cpu()),
                }
            )
            stats.update({k: float(v.cpu()) for k, v in stft_stats.items()})
            stats.update({k: float(v.cpu()) for k, v in sig_stats.items()})

        # Optimize
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        if gan_enabled and d_optimizer is not None and step >= gan_start:
            scaler.step(d_optimizer)
        scaler.update()

        step += 1
        if scheduler is not None:
            scheduler.step_batch(step)

        if step % int(cfg["train"]["log_interval_steps"]) == 0:
            row = {"step": step, **stats}
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
                scaler=scaler if scaler.is_enabled() else None,
                cfg=cfg,
                extra=_extra_state(),
            )

            # Optionally run probes on the just-saved checkpoint.
            if bool(cfg.get("eval", {}).get("enabled", False)) and bool(cfg["train"].get("run_eval_on_save", False)):
                from eval.run_probes import run_all_probes

                results = run_all_probes(
                    run_dir=str(out_root),
                    step=step,
                    exp_cfg=cfg,
                    ckpt_path=last_path,
                    python_bin=sys.executable,
                )
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

                    if emo.get("accuracy") is not None:
                        to_log["probe/emotion_accuracy"] = float(emo["accuracy"])
                    if emo.get("macro_f1") is not None:
                        to_log["probe/emotion_macro_f1"] = float(emo["macro_f1"])

                    if gen.get("accuracy") is not None:
                        to_log["probe/gender_accuracy"] = float(gen["accuracy"])

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
                        scaler=scaler if scaler.is_enabled() else None,
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
                        scaler=scaler if scaler.is_enabled() else None,
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
                        scaler=scaler if scaler.is_enabled() else None,
                        cfg=cfg,
                        extra=_extra_state(val=v),
                    )

            # Probes are triggered on save (run_eval_on_save), not on eval.

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
