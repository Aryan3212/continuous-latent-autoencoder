from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio

from schema import LossCfg, MelCfg, STFTCfg


def _get_window(kind: str, win_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kind == "hann":
        return torch.hann_window(win_length, device=device, dtype=dtype)
    raise ValueError(f"Unsupported window: {kind}")


def _stft_mag(
    x: torch.Tensor, n_fft: int, hop_length: int, win_length: int, *, center: bool, window: str
) -> torch.Tensor:
    # Precision boundary: the STFT / complex representation is computed in FP32 so
    # autocast bf16/fp16 never reaches torch.stft (the most dynamically ranged op).
    # Input + window are cast to float32 regardless of the surrounding autocast dtype.
    win = _get_window(window, win_length, device=x.device, dtype=torch.float32)
    x1 = x.float().squeeze(1)
    stft = torch.stft(
        x1,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=win,
        center=center,
        return_complex=True,
    )
    return stft.abs()


class MultiResSTFTLoss(nn.Module):
    def __init__(self, cfg: STFTCfg):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        return_per_sample: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x_hat.shape != x.shape:
            raise ValueError(f"x_hat and x must match; got {tuple(x_hat.shape)} vs {tuple(x.shape)}")
        
        batch_size = x.shape[0]
        per_sample_losses = torch.zeros(batch_size, device=x.device)
        
        # Stats accumulation
        total_sc = torch.zeros(batch_size, device=x.device)
        total_mag = torch.zeros(batch_size, device=x.device)
        total_log = torch.zeros(batch_size, device=x.device)
        
        for i_res, n_fft in enumerate(self.cfg.fft_sizes):
            hop = max(1, int(n_fft * self.cfg.hop_ratio))
            win = max(1, int(n_fft * self.cfg.win_ratio))
            mag_hat = _stft_mag(
                x_hat, n_fft=n_fft, hop_length=hop, win_length=win, center=self.cfg.center, window=self.cfg.window
            )
            
            mag = _stft_mag(
                x, n_fft=n_fft, hop_length=hop, win_length=win, center=self.cfg.center, window=self.cfg.window
            )

            # Reduce over Frequency (1) and Time (2) dimensions, keeping Batch (0)
            dims = (1, 2)

            # Spectral Convergence: Use unmasked denominator for stability
            denom = mag.norm(p="fro", dim=dims) + self.cfg.logmag_eps

            numer = (mag_hat - mag).norm(p="fro", dim=dims)
            sc = numer / denom
            
            # Log Mag
            l_mag = (mag_hat - mag).abs().mean(dim=dims)
            l_log = (torch.log(mag_hat + self.cfg.logmag_eps) - torch.log(mag + self.cfg.logmag_eps)).abs().mean(dim=dims)
            
            # Accumulate
            combined = self.cfg.sc_weight * sc + self.cfg.mag_weight * l_mag + self.cfg.logmag_weight * l_log
            per_sample_losses += combined
            
            total_sc += sc
            total_mag += l_mag
            total_log += l_log

        # Normalize by number of resolutions
        n_res = len(self.cfg.fft_sizes)
        per_sample_losses /= n_res
        total_sc /= n_res
        total_mag /= n_res
        total_log /= n_res

        if return_per_sample:
            # Return the (B,) tensor and the stats (B,) tensors
            stats = {
                "stft_loss": per_sample_losses,
                "stft_sc": total_sc,
                "stft_mag": total_mag,
                "stft_log": total_log,
            }
            return per_sample_losses, stats
        else:
            loss = per_sample_losses.mean()
            stats = {
                "stft_loss": loss.detach(),
                "stft_sc": total_sc.mean().detach(),
                "stft_mag": total_mag.mean().detach(),
                "stft_log": total_log.mean().detach(),
            }
            return loss, stats


class MelLoss(nn.Module):
    """Mel-spectrogram reconstruction loss.

    Computes STFT magnitudes, projects them onto a mel filterbank, then compares
    predicted vs target mel magnitudes with the same sc / mag / logmag weighting
    scheme as MultiResSTFTLoss. Because the mel representation is fundamentally a
    magnitude / log-magnitude quantity, this loss is, by construction, weighted
    toward mag / log_mag (spectral convergence is off by default).

    Returns stats prefixed with ``mel_*`` (mel_loss / mel_sc / mel_mag / mel_log)
    so it is interchangeable with MultiResSTFTLoss in the training loop and can be
    ablated against it.
    """

    def __init__(self, cfg: MelCfg, sample_rate: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        sr = int(sample_rate if sample_rate is not None else cfg.sample_rate)
        n_freqs = cfg.n_fft // 2 + 1
        fmax = float(cfg.fmax) if cfg.fmax is not None else sr / 2.0
        # (n_freqs, n_mels) mel filterbank; rebuilt on CPU, moved with .to(device).
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs, cfg.fmin, fmax, int(cfg.n_mels), sr
        )
        self.register_buffer("fb", fb, persistent=False)

    def _mel_mag(self, x: torch.Tensor) -> torch.Tensor:
        # STFT in FP32 (precision boundary) regardless of autocast dtype.
        win = _get_window(self.cfg.window, self.cfg.win_length, device=x.device, dtype=torch.float32)
        # x: (B,1,T) -> (B, n_freqs, T_time)
        spec = torch.stft(
            x.float().squeeze(1),
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=win,
            center=True,
            return_complex=True,
        )
        mag = spec.abs()
        # (B, T_time, n_freqs) @ (n_freqs, n_mels) -> (B, T_time, n_mels) -> (B, n_mels, T_time)
        fb = self.fb.to(device=mag.device, dtype=mag.dtype)
        mel = torch.matmul(mag.transpose(1, 2), fb).transpose(1, 2)
        return mel

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        return_per_sample: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x_hat.shape != x.shape:
            raise ValueError(f"x_hat and x must match; got {tuple(x_hat.shape)} vs {tuple(x.shape)}")

        batch_size = x.shape[0]
        per_sample_losses = torch.zeros(batch_size, device=x.device)
        total_sc = torch.zeros(batch_size, device=x.device)
        total_mag = torch.zeros(batch_size, device=x.device)
        total_log = torch.zeros(batch_size, device=x.device)

        mel_hat = self._mel_mag(x_hat)
        mel = self._mel_mag(x)
        dims = (1, 2)

        # Spectral convergence (off by default via sc_weight=0.0).
        denom = mel.norm(p="fro", dim=dims) + self.cfg.logmag_eps
        numer = (mel_hat - mel).norm(p="fro", dim=dims)
        sc = numer / denom

        l_mag = (mel_hat - mel).abs().mean(dim=dims)
        l_log = (
            torch.log(mel_hat + self.cfg.logmag_eps)
            - torch.log(mel + self.cfg.logmag_eps)
        ).abs().mean(dim=dims)

        combined = self.cfg.sc_weight * sc + self.cfg.mag_weight * l_mag + self.cfg.logmag_weight * l_log
        per_sample_losses += combined
        total_sc += sc
        total_mag += l_mag
        total_log += l_log

        if return_per_sample:
            stats = {
                "mel_loss": per_sample_losses,
                "mel_sc": total_sc,
                "mel_mag": total_mag,
                "mel_log": total_log,
            }
            return per_sample_losses, stats
        else:
            loss = per_sample_losses.mean()
            stats = {
                "mel_loss": loss.detach(),
                "mel_sc": total_sc.mean().detach(),
                "mel_mag": total_mag.mean().detach(),
                "mel_log": total_log.mean().detach(),
            }
            return loss, stats


class ReconSpectrogram(nn.Module):
    """Waveform -> magnitude spectrogram in the active reconstruction domain.

    Produces the SAME representation the reconstruction loss compares (mel or
    STFT magnitude), so the GAN discriminator can operate in that domain instead
    of on the raw waveform. Output shape is ``(B, F, T_frames)`` where ``F`` is the
    number of bins (``n_mels`` for mel, ``n_freqs`` for STFT); expose ``n_bins`` to
    size the discriminator's ``in_channels``.

    Mel mode reuses the MelLoss spectrogram recipe (STFT -> mel filterbank). STFT
    mode uses a single representative resolution (the first ``fft_sizes`` entry)
    so the discriminator sees a fixed-shape spectrogram.
    """

    def __init__(self, cfg: LossCfg, sample_rate: int):
        super().__init__()
        self.recon_type = cfg.recon_type
        sr = int(sample_rate)
        if cfg.recon_type == "mel":
            n_fft = cfg.mel.n_fft
            hop = cfg.mel.hop_length
            win = cfg.mel.win_length
            fmax = float(cfg.mel.fmax) if cfg.mel.fmax is not None else sr / 2.0
            fb = torchaudio.functional.melscale_fbanks(
                n_fft // 2 + 1, cfg.mel.fmin, fmax, int(cfg.mel.n_mels), sr
            )
            self.register_buffer("fb", fb, persistent=False)
            self.n_bins = cfg.mel.n_mels
        else:
            n_fft = cfg.stft.fft_sizes[0]
            hop = max(1, int(n_fft * cfg.stft.hop_ratio))
            win = max(1, int(n_fft * cfg.stft.win_ratio))
            self.n_bins = n_fft // 2 + 1
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.window = cfg.mel.window if cfg.recon_type == "mel" else cfg.stft.window

    def _spec_mag(self, wav: torch.Tensor) -> torch.Tensor:
        # STFT in FP32 (precision boundary) regardless of autocast dtype.
        win = _get_window(self.window, self.win, device=wav.device, dtype=torch.float32)
        # wav: (B,1,T) -> (B, n_freqs, T_frames)
        spec = torch.stft(
            wav.float().squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=win,
            center=True,
            return_complex=True,
        )
        mag = spec.abs()
        if self.recon_type == "mel":
            fb = self.fb.to(device=mag.device, dtype=mag.dtype)
            # (B, n_freqs, T) -> (B, n_mels, T)
            mag = torch.matmul(mag.transpose(1, 2), fb).transpose(1, 2)
        return mag

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self._spec_mag(wav)


# --------------------------------------------------------------------------- #
# Adversarial + feature-matching losses (HiFi-GAN multi-period discriminator).
# The discriminator outputs (d_real / d_fake) and feature maps are lists over
# sub-discriminators (one per period); see models/discriminator.py.
#
# Both the LSGAN and hinge objectives are supported via `loss_type`. They must
# NOT be combined: a single objective is retained throughout an experiment.
# Each sub-discriminator's contribution is MEAN-reduced over the logit elements
# (and batch); the per-branch terms are then averaged (torch.stack + mean) so the
# loss magnitude is independent of how many sub-discriminators (periods) exist —
# duplicating a discriminator scale must not silently multiply the objective or
# its gradient.
# --------------------------------------------------------------------------- #
def _zero_loss() -> torch.Tensor:
    """Explicit zero when no sub-discriminator outputs are present.

    `d_real`/`d_fake` are never empty in practice (MultiPeriodDiscriminator is
    always built from a non-empty period list), but this keeps the contract safe
    without indexing a non-existent tensor. Device follows the current CUDA device
    (or CPU) so it matches the training device.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.zeros((), device=device)


def _discriminator_loss_lsgan(
    d_real: List[torch.Tensor], d_fake: List[torch.Tensor]
) -> torch.Tensor:
    """LSGAN discriminator loss: pull real -> 1, fake -> 0."""
    if not d_real:
        return _zero_loss()
    losses = [((1.0 - dr).pow(2).mean() + dg.pow(2).mean()) for dr, dg in zip(d_real, d_fake)]
    return torch.stack(losses).mean()


def _discriminator_loss_hinge(
    d_real: List[torch.Tensor], d_fake: List[torch.Tensor]
) -> torch.Tensor:
    """Hinge discriminator loss: margin-based, no squaring of logits."""
    if not d_real:
        return _zero_loss()
    losses = [
        (torch.relu(1.0 - dr).mean() + torch.relu(1.0 + dg).mean())
        for dr, dg in zip(d_real, d_fake)
    ]
    return torch.stack(losses).mean()


def discriminator_loss(
    d_real: List[torch.Tensor], d_fake: List[torch.Tensor], loss_type: str = "lsgan"
) -> torch.Tensor:
    if loss_type == "lsgan":
        return _discriminator_loss_lsgan(d_real, d_fake)
    if loss_type == "hinge":
        return _discriminator_loss_hinge(d_real, d_fake)
    raise ValueError(f"Unsupported adv loss_type: {loss_type!r} (expected 'lsgan' or 'hinge')")


def _generator_adv_loss_lsgan(d_fake: List[torch.Tensor]) -> torch.Tensor:
    """LSGAN generator loss: push fake -> 1."""
    if not d_fake:
        return _zero_loss()
    losses = [(1.0 - dg).pow(2).mean() for dg in d_fake]
    return torch.stack(losses).mean()


def _generator_adv_loss_hinge(d_fake: List[torch.Tensor]) -> torch.Tensor:
    """Hinge generator loss: linear objective on the discriminator logits."""
    if not d_fake:
        return _zero_loss()
    losses = [-dg.mean() for dg in d_fake]
    return torch.stack(losses).mean()


def generator_adv_loss(d_fake: List[torch.Tensor], loss_type: str = "lsgan") -> torch.Tensor:
    if loss_type == "lsgan":
        return _generator_adv_loss_lsgan(d_fake)
    if loss_type == "hinge":
        return _generator_adv_loss_hinge(d_fake)
    raise ValueError(f"Unsupported adv loss_type: {loss_type!r} (expected 'lsgan' or 'hinge')")


def feature_matching_loss(
    fmap_real: List[List[torch.Tensor]], fmap_fake: List[List[torch.Tensor]]
) -> torch.Tensor:
    """Mean L1 between real/fake discriminator feature maps (real detached).

    Normalized by the TOTAL number of feature maps across every sub-discriminator
    and layer, so its magnitude is independent of discriminator topology (adding a
    scale or a layer cannot silently scale the objective/gradient).
    """
    loss = fmap_fake[0][0].new_zeros(())
    n = 0
    for fr_list, fg_list in zip(fmap_real, fmap_fake):
        for fr, fg in zip(fr_list, fg_list):
            loss = loss + (fr.detach() - fg).abs().mean()
            n += 1
    return loss / max(1, n)
