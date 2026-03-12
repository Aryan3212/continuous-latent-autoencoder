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
    logmag_eps: float = 1.0e-3
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

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor, return_per_sample: bool = False, target_mags: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x_hat.shape != x.shape:
            raise ValueError(f"x_hat and x must match; got {tuple(x_hat.shape)} vs {tuple(x.shape)}")
        
        batch_size = x.shape[0]
        per_sample_losses = torch.zeros(batch_size, device=x.device)
        
        # Stats accumulation
        total_sc = torch.zeros(batch_size, device=x.device)
        total_mag = torch.zeros(batch_size, device=x.device)
        total_log = torch.zeros(batch_size, device=x.device)
        
        for n_fft in self.cfg.fft_sizes:
            hop = max(1, int(n_fft * self.cfg.hop_ratio))
            win = max(1, int(n_fft * self.cfg.win_ratio))
            mag_hat = _stft_mag(
                x_hat, n_fft=n_fft, hop_length=hop, win_length=win, center=self.cfg.center, window=self.cfg.window
            )
            
            if target_mags is not None and str(n_fft) in target_mags:
                mag = target_mags[str(n_fft)].to(x.device, dtype=x.dtype)
            else:
                mag = _stft_mag(
                    x, n_fft=n_fft, hop_length=hop, win_length=win, center=self.cfg.center, window=self.cfg.window
                )
            
            # Reduce over Frequency (1) and Time (2) dimensions, keeping Batch (0)
            dims = (1, 2)
            
            # Spectral Convergence
            numer = (mag_hat - mag).norm(p="fro", dim=dims)
            denom = mag.norm(p="fro", dim=dims) + self.cfg.logmag_eps
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
