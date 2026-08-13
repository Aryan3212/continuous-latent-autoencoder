"""Shared per-utterance reconstruction metrics with explicit coverage records."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from losses import MultiResSTFTLoss
from schema import STFTCfg


METRIC_NAMES = (
    "stft_loss",
    "stft_sc",
    "stft_mag",
    "stft_log",
    "wav_l1",
    "si_sdr_db",
    "stoi",
    "estoi",
    "pesq_wb",
)
EVALUATION_STFT_CONFIG_VERSION = "tacl-reconstruction-v1"


def evaluation_stft_config() -> STFTCfg:
    """Paper-evaluation STFT contract shared by every reconstruction system."""
    return STFTCfg(
        fft_sizes=[256, 512, 1024, 2048],
        hop_ratio=0.25,
        win_ratio=1.0,
        center=True,
        window="hann",
        logmag_eps=1.0e-3,
        sc_weight=0.1,
        mag_weight=1.0,
        logmag_weight=1.0,
    )


def strict_coverage_failures(
    coverage: dict[str, dict[str, Any]], expected_items: int
) -> dict[str, dict[str, Any]]:
    """Return requested metrics that lack one successful score per source item."""
    return {
        metric: item
        for metric, item in coverage.items()
        if item.get("attempted") != expected_items
        or item.get("succeeded") != expected_items
        or item.get("failed") != 0
    }


def valid_num_samples(meta: dict[str, Any], sample_rate: int, padded_length: int) -> int:
    """Return the scored prefix length, honoring manifest duration when present."""
    duration = meta.get("duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        if math.isfinite(float(duration)) and float(duration) > 0:
            return min(padded_length, max(1, int(round(float(duration) * sample_rate))))
    return padded_length


def align_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    if waveform.size(-1) > target_length:
        return waveform[..., :target_length]
    if waveform.size(-1) < target_length:
        return torch.nn.functional.pad(waveform, (0, target_length - waveform.size(-1)))
    return waveform


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1.0e-8) -> float:
    """Zero-mean scale-invariant SDR in dB for two one-dimensional waveforms."""
    reference = reference.double() - reference.double().mean()
    estimate = estimate.double() - estimate.double().mean()
    reference_energy = reference.square().sum()
    if not torch.isfinite(reference_energy) or reference_energy <= eps:
        raise ValueError("reference is silent or non-finite after mean removal")
    projection = reference * (torch.dot(estimate, reference) / reference_energy)
    noise = estimate - projection
    ratio = projection.square().sum() / noise.square().sum().clamp_min(eps)
    if not torch.isfinite(ratio):
        raise ValueError("SI-SDR ratio is non-finite")
    return float(10.0 * torch.log10(ratio.clamp_min(eps)))


class ReconstructionMetrics:
    """Compute metrics per item and retain dependency/failure coverage."""

    def __init__(self, stft: MultiResSTFTLoss, sample_rate: int):
        self.stft = stft
        self.sample_rate = int(sample_rate)
        self.values: dict[str, list[float]] = defaultdict(list)
        self.attempted: dict[str, int] = defaultdict(int)
        self.failures: dict[str, list[str]] = defaultdict(list)
        self.dependencies: dict[str, dict[str, str]] = {}

        try:
            from pystoi import stoi as stoi_fn
        except (ImportError, OSError) as exc:
            self._stoi = None
            reason = f"pystoi unavailable: {type(exc).__name__}: {exc}"
            self.dependencies["stoi"] = {"status": "unavailable", "reason": reason}
            self.dependencies["estoi"] = {"status": "unavailable", "reason": reason}
        else:
            self._stoi = stoi_fn
            self.dependencies["stoi"] = {"status": "available", "package": "pystoi"}
            self.dependencies["estoi"] = {"status": "available", "package": "pystoi"}

        try:
            from pesq import pesq as pesq_fn
        except (ImportError, OSError) as exc:
            self._pesq = None
            self.dependencies["pesq_wb"] = {
                "status": "unavailable",
                "reason": f"pesq unavailable: {type(exc).__name__}: {exc}",
            }
        else:
            self._pesq = pesq_fn
            self.dependencies["pesq_wb"] = {"status": "available", "package": "pesq"}

    def _record(self, metric: str, fn: Any) -> float | None:
        self.attempted[metric] += 1
        try:
            value = float(fn())
            if not math.isfinite(value):
                raise ValueError(f"non-finite value {value}")
        except Exception as exc:  # Metrics have heterogeneous runtime errors.
            if len(self.failures[metric]) < 20:
                self.failures[metric].append(f"{type(exc).__name__}: {exc}")
            return None
        self.values[metric].append(value)
        return value

    @torch.no_grad()
    def evaluate(self, reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float | None]:
        """Score one aligned mono pair at ``sample_rate``."""
        reference = reference.detach().float().flatten()
        estimate = estimate.detach().float().flatten()
        if reference.numel() != estimate.numel() or reference.numel() == 0:
            raise ValueError("reference and estimate must have the same non-zero length")
        if not torch.isfinite(reference).all() or not torch.isfinite(estimate).all():
            raise ValueError("waveforms contain non-finite samples")

        result: dict[str, float | None] = {}
        minimum_stft_length = max(self.stft.cfg.fft_sizes)
        ref_stft = reference
        est_stft = estimate
        if reference.numel() < minimum_stft_length:
            pad = minimum_stft_length - reference.numel()
            ref_stft = torch.nn.functional.pad(reference, (0, pad))
            est_stft = torch.nn.functional.pad(estimate, (0, pad))
        _, stft_stats = self.stft(
            est_stft[None, None, :], ref_stft[None, None, :], return_per_sample=True
        )
        for key in ("stft_loss", "stft_sc", "stft_mag", "stft_log"):
            result[key] = self._record(key, lambda key=key: stft_stats[key][0].item())
        result["wav_l1"] = self._record(
            "wav_l1", lambda: torch.mean(torch.abs(estimate - reference)).item()
        )
        result["si_sdr_db"] = self._record(
            "si_sdr_db", lambda: si_sdr(reference.cpu(), estimate.cpu())
        )

        ref_np = reference.cpu().numpy().astype(np.float64, copy=False)
        est_np = estimate.cpu().numpy().astype(np.float64, copy=False)
        stoi_fn = self._stoi
        if stoi_fn is None:
            self.attempted["stoi"] += 1
            self.attempted["estoi"] += 1
            result["stoi"] = None
            result["estoi"] = None
        else:
            result["stoi"] = self._record(
                "stoi", lambda: stoi_fn(ref_np, est_np, self.sample_rate, extended=False)
            )
            result["estoi"] = self._record(
                "estoi", lambda: stoi_fn(ref_np, est_np, self.sample_rate, extended=True)
            )

        pesq_fn = self._pesq
        if pesq_fn is None:
            self.attempted["pesq_wb"] += 1
            result["pesq_wb"] = None
        elif self.sample_rate != 16_000:
            self.attempted["pesq_wb"] += 1
            if len(self.failures["pesq_wb"]) < 20:
                self.failures["pesq_wb"].append(
                    f"ValueError: wide-band PESQ requires 16000 Hz, got {self.sample_rate}"
                )
            result["pesq_wb"] = None
        else:
            result["pesq_wb"] = self._record(
                "pesq_wb", lambda: pesq_fn(16_000, ref_np, est_np, "wb")
            )
        return result

    def summary(self) -> dict[str, Any]:
        aggregate: dict[str, float | None] = {}
        coverage: dict[str, dict[str, Any]] = {}
        for metric in METRIC_NAMES:
            values = self.values.get(metric, [])
            attempted = self.attempted.get(metric, 0)
            aggregate[metric] = float(np.mean(values)) if values else None
            item: dict[str, Any] = {
                "attempted": attempted,
                "succeeded": len(values),
                "failed": max(0, attempted - len(values)),
            }
            dependency = self.dependencies.get(metric)
            if dependency is not None:
                item["dependency"] = dependency
            failures = self.failures.get(metric)
            if failures:
                item["failure_examples"] = failures
            coverage[metric] = item
        aggregate["stft"] = aggregate["stft_loss"]
        return {"aggregate": aggregate, "coverage": coverage}
