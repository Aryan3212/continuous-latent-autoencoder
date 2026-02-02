from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class MultiResSTFTConfig:
    fft_sizes: List[int]
    hop_ratio: float = 0.25
    win_ratio: float = 1.0
    center: bool = True
    window: str = "hann"
    logmag_eps: float = 1.0e-7
    sc_weight: float = 1.0
    mag_weight: float = 1.0
    logmag_weight: float = 1.0


def _get_window(kind: str, win_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kind == "hann":
        return torch.hann_window(win_length, device=device, dtype=dtype)
    raise ValueError(f"Unsupported window: {kind}")


def _stft_mag(
    x: torch.Tensor, n_fft: int, hop_length: int, win_length: int, *, center: bool, window: str
) -> torch.Tensor:
    win = _get_window(window, win_length, device=x.device, dtype=x.dtype)
    # x: (B,1,T) -> (B,T)
    x1 = x.squeeze(1)
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
    def __init__(self, cfg: MultiResSTFTConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x_hat.shape != x.shape:
            raise ValueError(f"x_hat and x must match; got {tuple(x_hat.shape)} vs {tuple(x.shape)}")
        losses = []
        sc_losses = []
        mag_losses = []
        log_losses = []
        for n_fft in self.cfg.fft_sizes:
            hop = max(1, int(n_fft * self.cfg.hop_ratio))
            win = max(1, int(n_fft * self.cfg.win_ratio))
            mag_hat = _stft_mag(
                x_hat, n_fft=n_fft, hop_length=hop, win_length=win, center=self.cfg.center, window=self.cfg.window
            )
            mag = _stft_mag(
                x, n_fft=n_fft, hop_length=hop, win_length=win, center=self.cfg.center, window=self.cfg.window
            )
            sc = (mag_hat - mag).norm(p="fro") / (mag.norm(p="fro") + self.cfg.logmag_eps)
            l_mag = (mag_hat - mag).abs().mean()
            l_log = (torch.log(mag_hat + self.cfg.logmag_eps) - torch.log(mag + self.cfg.logmag_eps)).abs().mean()
            sc_losses.append(sc)
            mag_losses.append(l_mag)
            log_losses.append(l_log)
            losses.append(
                self.cfg.sc_weight * sc + self.cfg.mag_weight * l_mag + self.cfg.logmag_weight * l_log
            )
        loss = torch.stack(losses).mean()
        stats = {
            "stft_loss": loss.detach(),
            "stft_sc": torch.stack(sc_losses).mean().detach(),
            "stft_mag": torch.stack(mag_losses).mean().detach(),
            "stft_log": torch.stack(log_losses).mean().detach(),
        }
        return loss, stats
