#!/usr/bin/env python3
"""Create and verify uncompressed TAR shards from an existing audio JSONL manifest.

The manifest is deliberately the sole inventory for this tool.  It never walks a
dataset directory, changes a split, or writes a loose canonical-audio staging
tree.  The output is intended to be consumed by a future streaming loader; this
script does not modify the current file-backed training path.

Run on the machine that has the dataset and the project's audio dependencies::

    uv run python scripts/prepare_audio_shards.py pack \
        --manifest staging/manifests/train.jsonl \
        --output-dir staging/packed/train --workers 4 --resume
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import os
import pathlib
import random
import tarfile
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import torch
import torchaudio


FORMAT_VERSION = 1
STATE_FILENAME = ".packing_state.json"
PARTS_DIRNAME = ".index_parts"
SHARDS_DIRNAME = "shards"
DESCRIPTOR_FILENAME = "shard_manifest.json"
INDEX_FILENAME = "index.jsonl"
EPSILON = 1.0e-30
MIN_TARGET_SHARD_BYTES = int(0.5 * 1024**3)
MAX_SHARD_BYTES = 2 * 1024**3
# tarfile closes an archive with two zero blocks and record padding.  Reserve a
# full record, not merely the two blocks, when enforcing the hard 2 GiB cap.
TAR_FINAL_RECORD_BYTES = tarfile.RECORDSIZE
MAX_ENCODE_WORKERS = 8
# Keep explicit headroom below full scale, rather than depending on a backend's
# treatment of exactly +1.0.  0.1% is far larger than float roundoff but far
# below any meaningful audio-level change once the loader restores the gain.
PCM16_STORAGE_PEAK = 0.999
SCALING_EXAMPLE_LIMIT = 3


class PackingError(RuntimeError):
    """A malformed manifest row or an unsafe canonicalization result."""


@dataclass(frozen=True)
class InputRecord:
    """One manifest row, with the same relative-path resolution as training."""

    row: dict[str, Any]
    source_path: pathlib.Path
    sample_id: str
    source_dataset: str


@dataclass
class EncodedSample:
    record: InputRecord
    audio: bytes
    metadata: bytes
    frame_count: int
    duration_seconds: float
    quantization: dict[str, float | int]
    amplitude_restore_gain: float
    canonical_peak: float
    storage_peak: float


def _strict_json_bytes(value: Any) -> bytes:
    """Serialize metadata deterministically and reject non-standard JSON numbers."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write_bytes(path: pathlib.Path, payload: bytes) -> None:
    """Publish a complete small metadata file with a same-directory rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as file:
        tmp = pathlib.Path(file.name)
        try:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def _atomic_write_json(path: pathlib.Path, value: Any) -> None:
    _atomic_write_bytes(path, _strict_json_bytes(value) + b"\n")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_manifest_root(manifest_path: pathlib.Path, items: list[dict[str, Any]]) -> pathlib.Path:
    """Match ``data_loading.resolve_manifest_root`` exactly for one manifest."""
    parent = manifest_path.resolve().parent
    for item in items:
        raw_path = item.get("audio_filepath")
        if not raw_path or os.path.isabs(str(raw_path)):
            continue
        for candidate in (parent, parent.parent):
            if (candidate / str(raw_path)).exists():
                return candidate
        raise FileNotFoundError(
            f"relative audio_filepath {raw_path!r} from {manifest_path} not found "
            f"under {parent} or {parent.parent}"
        )
    return parent


def _existing_id(row: dict[str, Any]) -> str | None:
    """Use the conventional manifest ID without guessing at dataset-specific fields."""
    value = row.get("id")
    if value is None or str(value) == "":
        return None
    return str(value)


def _sample_id(row: dict[str, Any]) -> tuple[str, str]:
    dataset = str(row.get("dataset") or "unknown")
    existing = _existing_id(row)
    if existing is not None:
        return f"{dataset}:{existing}", dataset
    # The whole original row is part of the fallback.  It is stable for this
    # authoritative manifest and deliberately turns duplicate anonymous rows
    # into a collision instead of silently giving them different identities.
    fallback = hashlib.sha256(_strict_json_bytes(row)).hexdigest()
    return f"{dataset}:sha256:{fallback}", dataset


def read_inventory(manifest_path: pathlib.Path) -> list[InputRecord]:
    """Read the already-combined manifest and reject ambiguous inventory rows."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PackingError(f"invalid JSON in {manifest_path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise PackingError(f"manifest row {line_number} is not a JSON object")
            if not row.get("audio_filepath"):
                raise PackingError(f"manifest row {line_number} lacks audio_filepath")
            # Detect NaN/Infinity and other values that cannot be preserved in
            # adjacent JSON metadata before any output is published.
            _strict_json_bytes(row)
            rows.append(row)
    if not rows:
        raise PackingError(f"manifest contains no records: {manifest_path}")

    root = resolve_manifest_root(manifest_path, rows)
    inventory: list[InputRecord] = []
    seen_ids: set[str] = set()
    seen_member_stems: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        raw_path = str(row["audio_filepath"])
        source_path = pathlib.Path(raw_path)
        if not source_path.is_absolute():
            source_path = root / source_path
        sample_id, dataset = _sample_id(row)
        if sample_id in seen_ids:
            raise PackingError(
                f"duplicate stable sample ID {sample_id!r} at manifest row {row_number}; "
                "supply unique dataset + id values"
            )
        seen_ids.add(sample_id)
        member_stem = _member_stem(sample_id)
        if member_stem in seen_member_stems:
            raise PackingError(
                f"hashed TAR member-key collision for stable sample ID {sample_id!r}; refusing overwrite"
            )
        seen_member_stems.add(member_stem)
        inventory.append(
            InputRecord(row=row, source_path=source_path, sample_id=sample_id, source_dataset=dataset)
        )
    return inventory


def _member_stem(sample_id: str) -> str:
    """An opaque safe member key, independent of unsafe user-supplied IDs."""
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def _empty_quantization() -> dict[str, float | int]:
    return {
        "sample_count": 0,
        "frame_count": 0,
        "max_abs_error": 0.0,
        "sum_squared_error": 0.0,
        "sum_squared_signal": 0.0,
    }


def _empty_amplitude_scaling() -> dict[str, float | int]:
    """Aggregate reversible PCM16 storage scaling without changing audio semantics."""
    return {
        "scaled_sample_count": 0,
        "max_restore_gain": 1.0,
        "max_canonical_peak": 0.0,
    }


def _merge_amplitude_scaling(
    aggregate: dict[str, float | int], sample: dict[str, float | int]
) -> dict[str, float | int]:
    return {
        "scaled_sample_count": int(aggregate["scaled_sample_count"])
        + int(sample["scaled_sample_count"]),
        "max_restore_gain": max(
            float(aggregate["max_restore_gain"]), float(sample["max_restore_gain"])
        ),
        "max_canonical_peak": max(
            float(aggregate["max_canonical_peak"]), float(sample["max_canonical_peak"])
        ),
    }


def _sample_amplitude_scaling(
    amplitude_restore_gain: float, canonical_peak: float
) -> dict[str, float | int]:
    return {
        "scaled_sample_count": int(amplitude_restore_gain > 1.0),
        "max_restore_gain": amplitude_restore_gain,
        "max_canonical_peak": canonical_peak,
    }


def _same_amplitude_scaling(
    left: dict[str, float | int], right: dict[str, float | int]
) -> bool:
    return (
        int(left["scaled_sample_count"]) == int(right["scaled_sample_count"])
        and math.isclose(
            float(left["max_restore_gain"]),
            float(right["max_restore_gain"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(left["max_canonical_peak"]),
            float(right["max_canonical_peak"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _amplitude_restore_gain(canonical_peak: float) -> float:
    """Return the reversible gain needed to keep PCM16 storage below full scale."""
    if not math.isfinite(canonical_peak) or canonical_peak < 0.0:
        raise PackingError(f"invalid canonical peak for PCM16 storage scaling: {canonical_peak!r}")
    if canonical_peak <= PCM16_STORAGE_PEAK:
        return 1.0
    # A deliberately larger-than-roundoff cushion keeps float32 source values
    # below the stated storage peak as well as float64 values.  Storage itself
    # is calculated in float64 below, so this cannot be lost by a Torch scalar
    # cast.  The gain is reversed by the packed loader.
    return math.nextafter(
        (canonical_peak / PCM16_STORAGE_PEAK) * (1.0 + 1.0e-6), math.inf
    )


def _optional_restore_gain(value: Any, label: str) -> float:
    """Read a legacy-optional gain while rejecting malformed new metadata."""
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackingError(f"{label} must be a finite numeric value >= 1.0")
    gain = float(value)
    if not math.isfinite(gain) or gain < 1.0:
        raise PackingError(f"{label} must be a finite numeric value >= 1.0")
    return gain


def _optional_nonnegative_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackingError(f"{label} must be a finite non-negative numeric value")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PackingError(f"{label} must be a finite non-negative numeric value")
    return number


def _merge_quantization(
    aggregate: dict[str, float | int], sample: dict[str, float | int]
) -> dict[str, float | int]:
    merged = dict(aggregate)
    merged["sample_count"] = int(merged["sample_count"]) + int(sample["sample_count"])
    merged["frame_count"] = int(merged["frame_count"]) + int(sample["frame_count"])
    merged["max_abs_error"] = max(float(merged["max_abs_error"]), float(sample["max_abs_error"]))
    merged["sum_squared_error"] = float(merged["sum_squared_error"]) + float(
        sample["sum_squared_error"]
    )
    merged["sum_squared_signal"] = float(merged["sum_squared_signal"]) + float(
        sample["sum_squared_signal"]
    )
    return merged


def _finalize_quantization(raw: dict[str, float | int]) -> dict[str, float | int | None]:
    frames = int(raw["frame_count"])
    error_energy = float(raw["sum_squared_error"])
    signal_energy = float(raw["sum_squared_signal"])
    error_rms = math.sqrt(error_energy / frames) if frames else 0.0
    signal_rms = math.sqrt(signal_energy / frames) if frames else 0.0
    snr_db: float | None
    if error_energy == 0.0:
        snr_db = None
    else:
        snr_db = 10.0 * math.log10(max(signal_energy, EPSILON) / error_energy)
    return {
        "sample_count": int(raw["sample_count"]),
        "frame_count": frames,
        "max_abs_error": float(raw["max_abs_error"]),
        "rms_error": error_rms,
        "signal_rms": signal_rms,
        "snr_db": snr_db,
        "sum_squared_error": error_energy,
        "sum_squared_signal": signal_energy,
    }


def _encode_one(record: InputRecord, sample_rate: int) -> EncodedSample:
    """Canonicalize one complete utterance using the current loader's order."""
    try:
        waveform, source_rate = torchaudio.load(str(record.source_path))
    except Exception as exc:  # pragma: no cover - exercised on remote audio files
        raise PackingError(
            f"failed to decode sample {record.sample_id!r} from {record.source_path}: {exc}"
        ) from exc
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.flatten()
    if int(source_rate) != sample_rate:
        waveform = torchaudio.transforms.Resample(int(source_rate), sample_rate)(waveform)
    waveform = waveform.contiguous()
    if waveform.numel() == 0:
        raise PackingError(f"empty waveform for sample {record.sample_id!r} ({record.source_path})")
    if not bool(torch.isfinite(waveform).all().item()):
        raise PackingError(f"non-finite canonical waveform for {record.sample_id!r} ({record.source_path})")
    canonical_peak = float(waveform.abs().max().item())
    canonical = waveform.detach().cpu().numpy().astype(np.float64, copy=False)
    amplitude_restore_gain = _amplitude_restore_gain(canonical_peak)
    # Store a uniformly scaled copy only when PCM16 needs extra headroom.  The
    # loader restores this gain before crop/pad/augmentation, so this is not
    # loudness normalization and does not alter the effective training sample.
    storage = canonical / amplitude_restore_gain
    storage_peak = float(np.max(np.abs(storage)))
    if storage_peak > PCM16_STORAGE_PEAK:
        raise PackingError(
            f"internal PCM16 storage scaling failed for {record.sample_id!r}: "
            f"storage peak {storage_peak:.9g} exceeds {PCM16_STORAGE_PEAK:.9g}"
        )
    encoded_buffer = io.BytesIO()
    try:
        sf.write(encoded_buffer, storage, sample_rate, format="FLAC", subtype="PCM_16")
        encoded = encoded_buffer.getvalue()
        # Decode what was encoded, rather than estimating error from an assumed
        # PCM mapping.  This catches encoder/backend changes as well.
        decoded, decoded_rate = sf.read(io.BytesIO(encoded), dtype="float64", always_2d=False)
        info = sf.info(io.BytesIO(encoded))
    except Exception as exc:  # pragma: no cover - backend failure is remote-specific
        raise PackingError(f"failed to encode PCM16 FLAC for {record.sample_id!r}: {exc}") from exc
    decoded = np.asarray(decoded, dtype=np.float64).reshape(-1)
    if (
        int(decoded_rate) != sample_rate
        or int(info.samplerate) != sample_rate
        or int(info.channels) != 1
        or str(info.subtype) != "PCM_16"
        or decoded.shape != canonical.shape
    ):
        raise PackingError(
            f"FLAC round-trip contract failed for {record.sample_id!r}: "
            f"rate={info.samplerate}, channels={info.channels}, subtype={info.subtype}, "
            f"decoded_frames={decoded.size}, expected_frames={canonical.size}"
        )
    restored = decoded * amplitude_restore_gain
    difference = restored - canonical
    squared_error = float(np.dot(difference, difference))
    squared_signal = float(np.dot(canonical, canonical))
    quantization = {
        "sample_count": 1,
        "frame_count": int(canonical.size),
        "max_abs_error": float(np.max(np.abs(difference))),
        "sum_squared_error": squared_error,
        "sum_squared_signal": squared_signal,
    }
    # Keep the source row under its own key rather than merging packed fields
    # into it: a dataset is allowed to have a field named ``packed`` (or any
    # future canonical-field name), and no original metadata may be overwritten.
    packed_metadata = {
        "original": record.row,
        "packed": {
            "sample_id": record.sample_id,
            "source_dataset": record.source_dataset,
            "source_path": str(record.source_path),
            "canonical_sample_rate": sample_rate,
            "canonical_channels": 1,
            "canonical_subtype": "PCM_16",
            "canonical_frame_count": int(canonical.size),
            "canonical_duration_seconds": float(canonical.size / sample_rate),
            "canonical_peak": canonical_peak,
            "storage_peak": storage_peak,
            "amplitude_restore_gain": amplitude_restore_gain,
        },
    }
    return EncodedSample(
        record=record,
        audio=encoded,
        metadata=_strict_json_bytes(packed_metadata) + b"\n",
        frame_count=int(canonical.size),
        duration_seconds=float(canonical.size / sample_rate),
        quantization=quantization,
        amplitude_restore_gain=amplitude_restore_gain,
        canonical_peak=canonical_peak,
        storage_peak=storage_peak,
    )


def _tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    # Normalize archive metadata so rerunning the same inventory/config produces
    # stable TAR members independent of host user and source file mtimes.
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def _estimated_member_bytes(sample: EncodedSample) -> int:
    # TAR writes one 512-byte header and pads every member to a 512-byte block.
    def member_size(payload: bytes) -> int:
        return 512 + ((len(payload) + 511) // 512) * 512

    return member_size(sample.audio) + member_size(sample.metadata)


def _write_index_part(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(_strict_json_bytes(row) + b"\n" for row in rows)
    _atomic_write_bytes(path, payload)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise PackingError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackingError(f"expected JSON object in {path}")
    return value


def _state_contract(
    manifest_sha256: str,
    sample_rate: int,
    target_shard_size_bytes: int,
    seed: int,
    record_count: int,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "manifest_sha256": manifest_sha256,
        "sample_rate": sample_rate,
        "target_shard_size_bytes": target_shard_size_bytes,
        "seed": seed,
        "record_count": record_count,
        "preprocessing": _current_preprocessing_contract(),
    }


def _legacy_preprocessing_contract() -> dict[str, str]:
    """The published v1 contract before reversible PCM16 storage scaling."""
    return {
        "decode": "torchaudio.load",
        "channel_conversion": "mean channels to mono before resampling",
        "resample": "torchaudio.transforms.Resample(source_rate, target_rate) defaults",
        "crop_or_pad": "none; full utterance is stored",
        "normalization": "none",
        "silence_removal": "none",
        "filtering": "none",
        "output": "FLAC PCM_16, mono, target sample rate",
        "out_of_range_policy": "fail if canonical absolute peak exceeds 1.0; never clip",
    }


def _current_preprocessing_contract() -> dict[str, str]:
    return {
        **_legacy_preprocessing_contract(),
        "out_of_range_policy": (
            "reversible per-sample storage scaling below PCM16 full scale; never clip"
        ),
        "storage_peak_limit": f"{PCM16_STORAGE_PEAK:.9g}",
    }


def _new_state(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        **contract,
        "next_record": 0,
        "next_shard_number": 0,
        "active_shard": None,
        "shards": [],
    }


def _resume_preprocessing_version(state: dict[str, Any], contract: dict[str, Any]) -> str:
    for field in (
        "format_version",
        "manifest_sha256",
        "sample_rate",
        "target_shard_size_bytes",
        "seed",
        "record_count",
    ):
        if state.get(field) != contract.get(field):
            raise PackingError(
                f"cannot resume: {field} differs from the interrupted packing state; "
                "use the original manifest and pack settings, or choose a new output directory"
            )
    preprocessing = state.get("preprocessing")
    if preprocessing == contract["preprocessing"]:
        return "current"
    if preprocessing == _legacy_preprocessing_contract():
        return "legacy"
    raise PackingError(
        "cannot resume: preprocessing differs from the supported packing contracts; "
        "use a new output directory"
    )


def _normalise_legacy_scaling_stats(shard: dict[str, Any]) -> None:
    """Add zero/one defaults for v1 shards that predate storage scaling."""
    scaling = shard.get("amplitude_scaling")
    if scaling is None:
        shard["amplitude_scaling"] = _empty_amplitude_scaling()
        return
    if not isinstance(scaling, dict):
        raise PackingError("invalid amplitude_scaling in interrupted packing state")
    try:
        scaled_count = int(scaling["scaled_sample_count"])
        max_gain = float(scaling["max_restore_gain"])
        max_peak = float(scaling["max_canonical_peak"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PackingError("invalid amplitude_scaling in interrupted packing state") from exc
    if scaled_count < 0 or not math.isfinite(max_gain) or max_gain < 1.0 or not math.isfinite(max_peak):
        raise PackingError("invalid amplitude_scaling in interrupted packing state")


def _migrate_legacy_interrupted_state(state: dict[str, Any], contract: dict[str, Any]) -> None:
    """Upgrade the known v1 failure-on-peak state without redoing finalized shards."""
    shards = state.get("shards")
    if not isinstance(shards, list):
        raise PackingError("invalid shards in interrupted packing state")
    for shard in shards:
        if not isinstance(shard, dict):
            raise PackingError("invalid shard in interrupted packing state")
        _normalise_legacy_scaling_stats(shard)
    state["preprocessing"] = contract["preprocessing"]
    print(
        "[packed-shards] migrating interrupted legacy v1 state to reversible PCM16 "
        "storage scaling; finalized shards remain unchanged",
        flush=True,
    )


def _remove_active_partial(output_dir: pathlib.Path, state: dict[str, Any]) -> None:
    """Remove only the named artifact that this state recorded as incomplete."""
    active = state.get("active_shard")
    if not active:
        return
    if not isinstance(active, dict) or not isinstance(active.get("name"), str):
        raise PackingError("invalid active_shard in packing state")
    name = active["name"]
    if pathlib.PurePosixPath(name).name != name or not name.startswith("train-"):
        raise PackingError("unsafe active_shard name in packing state")
    # A crash can happen after either of these publications but before the state
    # transaction.  Both are producer-owned paths explicitly named in state.
    for path in (
        output_dir / SHARDS_DIRNAME / f"{name}.tar",
        output_dir / PARTS_DIRNAME / f"{name}.index.jsonl",
    ):
        path.unlink(missing_ok=True)
    for path in (
        output_dir / SHARDS_DIRNAME / f".{name}.tar.tmp",
        output_dir / PARTS_DIRNAME / f".{name}.index.jsonl.tmp",
    ):
        path.unlink(missing_ok=True)
    state["active_shard"] = None


def _initialize_output(
    output_dir: pathlib.Path, contract: dict[str, Any], resume: bool
) -> dict[str, Any]:
    state_path = output_dir / STATE_FILENAME
    descriptor_path = output_dir / DESCRIPTOR_FILENAME
    if output_dir.exists() and not output_dir.is_dir():
        raise PackingError(f"output path is not a directory: {output_dir}")
    if resume:
        # Case 1: the final descriptor is a complete-publication marker.  It
        # must still describe this exact manifest/packing contract; otherwise a
        # populated directory is unrelated and is never reused.
        if descriptor_path.exists():
            descriptor = _load_json(descriptor_path)
            version = _resume_preprocessing_version(descriptor, contract)
            if version == "legacy":
                print(
                    "[packed-shards] existing legacy v1 output has no reversible scaling metadata; "
                    "verifying it unchanged with restore gain defaulting to 1.0",
                    flush=True,
                )
            return {"complete": True, "descriptor": descriptor}

        # Case 2: a matching state file means a prior pack was interrupted.
        # Reconcile only the producer-owned active shard named by that state.
        if state_path.is_file():
            state = _load_json(state_path)
            version = _resume_preprocessing_version(state, contract)
            if version == "legacy":
                _migrate_legacy_interrupted_state(state, contract)
            _remove_active_partial(output_dir, state)
            _atomic_write_json(state_path, state)
            return state

        # Case 3: ``--resume`` is intentionally start-or-resume.  An absent or
        # empty output directory starts a fresh run, while any other nonempty
        # directory has no matching state and is rejected as unrelated.
        if output_dir.exists() and any(output_dir.iterdir()):
            raise PackingError(
                f"--resume found no producer state in nonempty output directory {output_dir}; "
                "refusing to reuse unrelated files"
            )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PackingError(
            f"output directory is nonempty: {output_dir}; use a new directory or --resume "
            "for a matching interrupted packing run"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SHARDS_DIRNAME).mkdir(exist_ok=True)
    (output_dir / PARTS_DIRNAME).mkdir(exist_ok=True)
    state = _new_state(contract)
    _atomic_write_json(state_path, state)
    return state


def _start_shard(
    output_dir: pathlib.Path, state: dict[str, Any]
) -> tuple[str, pathlib.Path, pathlib.Path, tarfile.TarFile]:
    number = int(state["next_shard_number"])
    name = f"train-{number:06d}"
    tar_path = output_dir / SHARDS_DIRNAME / f"{name}.tar"
    temp_path = output_dir / SHARDS_DIRNAME / f".{name}.tar.tmp"
    part_path = output_dir / PARTS_DIRNAME / f"{name}.index.jsonl"
    if tar_path.exists() or temp_path.exists() or part_path.exists():
        raise PackingError(f"refusing to overwrite existing shard artifact for {name}")
    state["active_shard"] = {"name": name}
    _atomic_write_json(output_dir / STATE_FILENAME, state)
    archive = tarfile.open(temp_path, mode="w", format=tarfile.PAX_FORMAT)
    return name, tar_path, part_path, archive


def _finalize_shard(
    output_dir: pathlib.Path,
    state: dict[str, Any],
    name: str,
    tar_path: pathlib.Path,
    part_path: pathlib.Path,
    archive: tarfile.TarFile,
    index_rows: list[dict[str, Any]],
    payload_bytes: int,
    duration_seconds: float,
    quantization: dict[str, float | int],
    amplitude_scaling: dict[str, float | int],
) -> None:
    temp_path = output_dir / SHARDS_DIRNAME / f".{name}.tar.tmp"
    archive.close()
    # fsync before publication so the state never describes a partially flushed TAR.
    with temp_path.open("rb") as file:
        os.fsync(file.fileno())
    tar_bytes = temp_path.stat().st_size
    # The estimate reserves a full final TAR record, but assert the post-close
    # physical file too.  This protects the hard cap if tarfile's implementation
    # or metadata behavior changes; active state keeps this unpublished shard
    # safely redoable on a later start-or-resume invocation.
    if tar_bytes > MAX_SHARD_BYTES:
        raise PackingError(
            f"finalized shard {name} is {tar_bytes} bytes, above the hard 2 GiB cap; "
            "reduce the target size or split the oversized source explicitly"
        )
    _write_index_part(part_path, index_rows)
    os.replace(temp_path, tar_path)
    shard = {
        "name": name,
        "path": f"{SHARDS_DIRNAME}/{tar_path.name}",
        "index_part": f"{PARTS_DIRNAME}/{part_path.name}",
        "count": len(index_rows),
        "audio_payload_bytes": payload_bytes,
        "tar_bytes": tar_bytes,
        "duration_seconds": duration_seconds,
        "quantization": quantization,
        "amplitude_scaling": amplitude_scaling,
    }
    state["shards"].append(shard)
    state["next_record"] = int(state["next_record"]) + len(index_rows)
    state["next_shard_number"] = int(state["next_shard_number"]) + 1
    state["active_shard"] = None
    _atomic_write_json(output_dir / STATE_FILENAME, state)


class OrderedEncoder:
    """Bounded, ordered prefetch of independent CPU decode/encode jobs."""

    def __init__(self, records: list[InputRecord], start: int, sample_rate: int, workers: int):
        self.records = records
        self.position = start
        self.sample_rate = sample_rate
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.futures: dict[int, concurrent.futures.Future[EncodedSample]] = {}
        # The executor itself is bounded by the CLI cap; this separate limit is
        # intentionally explicit because every completed future owns a FLAC
        # byte buffer until its ordered predecessor has been written.
        self.max_in_flight = min(workers, MAX_ENCODE_WORKERS)
        self._submit_until_full()

    def _submit_until_full(self) -> None:
        while len(self.futures) < self.max_in_flight and self.position < len(self.records):
            index = self.position
            self.futures[index] = self.executor.submit(_encode_one, self.records[index], self.sample_rate)
            self.position += 1

    def __iter__(self) -> Iterator[EncodedSample]:
        next_index = min(self.futures) if self.futures else self.position
        try:
            while self.futures:
                future = self.futures.pop(next_index)
                sample = future.result()
                self._submit_until_full()
                next_index += 1
                yield sample
        finally:
            for future in self.futures.values():
                future.cancel()
            self.executor.shutdown(wait=True, cancel_futures=True)


def _build_final_index(output_dir: pathlib.Path, shards: list[dict[str, Any]]) -> None:
    output_path = output_dir / INDEX_FILENAME
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_dir, prefix=f".{INDEX_FILENAME}.", suffix=".tmp", delete=False
    ) as destination:
        tmp_path = pathlib.Path(destination.name)
        try:
            for shard in shards:
                part_path = output_dir / shard["index_part"]
                with part_path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    os.replace(tmp_path, output_path)


def _build_descriptor(contract: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    # ``index_part`` is an interrupted-pack implementation detail.  It remains
    # in state until publication, but a completed public descriptor does not
    # require it and must be usable after those temporary files are cleaned up.
    shards = [
        {key: value for key, value in shard.items() if key != "index_part"}
        for shard in state["shards"]
    ]
    raw_quantization = _empty_quantization()
    raw_amplitude_scaling = _empty_amplitude_scaling()
    for shard in shards:
        raw_quantization = _merge_quantization(raw_quantization, shard["quantization"])
        raw_amplitude_scaling = _merge_amplitude_scaling(
            raw_amplitude_scaling,
            _shard_amplitude_scaling(shard),
        )
    return {
        **contract,
        "kind": "continuous-latent-ae-audio-tar-shards",
        "shard_order": "deterministic random permutation of authoritative manifest rows",
        "shard_container": "uncompressed POSIX TAR; FLAC members are individually compressed",
        "member_layout": "samples/<sha256(sample_id)>.flac and adjacent .json",
        "counts": {
            "samples": int(state["next_record"]),
            "shards": len(shards),
            "total_duration_seconds": sum(float(shard["duration_seconds"]) for shard in shards),
            "total_tar_bytes": sum(int(shard["tar_bytes"]) for shard in shards),
            "total_audio_payload_bytes": sum(int(shard["audio_payload_bytes"]) for shard in shards),
        },
        "quantization": _finalize_quantization(raw_quantization),
        "amplitude_scaling": raw_amplitude_scaling,
        "shards": shards,
        "index": INDEX_FILENAME,
    }


def _shard_amplitude_scaling(shard: dict[str, Any]) -> dict[str, float | int]:
    """Legacy finalized v1 shards have no scaling fields and imply gain 1.0."""
    scaling = shard.get("amplitude_scaling")
    if scaling is None:
        return _empty_amplitude_scaling()
    if not isinstance(scaling, dict):
        raise PackingError("invalid amplitude_scaling in shard metadata")
    try:
        normalized = {
            "scaled_sample_count": int(scaling["scaled_sample_count"]),
            "max_restore_gain": float(scaling["max_restore_gain"]),
            "max_canonical_peak": float(scaling["max_canonical_peak"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise PackingError("invalid amplitude_scaling in shard metadata") from exc
    if (
        normalized["scaled_sample_count"] < 0
        or not math.isfinite(float(normalized["max_restore_gain"]))
        or float(normalized["max_restore_gain"]) < 1.0
        or not math.isfinite(float(normalized["max_canonical_peak"]))
        or float(normalized["max_canonical_peak"]) < 0.0
    ):
        raise PackingError("invalid amplitude_scaling in shard metadata")
    return normalized


def pack(args: argparse.Namespace) -> None:
    manifest_path = pathlib.Path(args.manifest).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    if args.sample_rate <= 0:
        raise PackingError("--sample-rate must be positive")
    if not 0.5 <= args.target_shard_size_gb <= 2.0:
        raise PackingError("--target-shard-size-gb must be between 0.5 and 2.0 GiB")
    if not 1 <= args.workers <= MAX_ENCODE_WORKERS:
        raise PackingError(f"--workers must be between 1 and {MAX_ENCODE_WORKERS}")
    if args.progress_interval_seconds <= 0.0:
        raise PackingError("--progress-interval-seconds must be positive")
    inventory = read_inventory(manifest_path)
    # A deterministic global shuffle mixes source datasets, speakers, and
    # durations statistically without reading audio twice or changing the split.
    random.Random(args.seed).shuffle(inventory)
    target_bytes = int(args.target_shard_size_gb * 1024**3)
    if not MIN_TARGET_SHARD_BYTES <= target_bytes <= MAX_SHARD_BYTES:
        raise PackingError("target shard byte count is outside the supported 0.5–2 GiB range")
    contract = _state_contract(
        manifest_sha256=_sha256_file(manifest_path),
        sample_rate=args.sample_rate,
        target_shard_size_bytes=target_bytes,
        seed=args.seed,
        record_count=len(inventory),
    )
    state = _initialize_output(output_dir, contract, args.resume)
    if state.get("complete"):
        verify_output(output_dir)
        print(f"[packed-shards] existing verified output: {output_dir}", flush=True)
        return

    start = int(state["next_record"])
    if start < 0 or start > len(inventory):
        raise PackingError("invalid next_record in interrupted packing state")
    # Each bounded encode worker may invoke Torch kernels during resampling.  A
    # one-thread intra-op pool prevents four workers from multiplying into the
    # host's full CPU thread count.  This is set before any worker is created.
    torch.set_num_threads(1)
    prior_duration_seconds = sum(float(shard["duration_seconds"]) for shard in state["shards"])
    prior_scaling = _empty_amplitude_scaling()
    for shard in state["shards"]:
        prior_scaling = _merge_amplitude_scaling(prior_scaling, _shard_amplitude_scaling(shard))
    print(
        "[packed-shards] start: "
        f"records={len(inventory)}, resume_record={start}, workers={args.workers}, "
        f"target_shard_gib={target_bytes / 1024**3:.3f}, "
        f"manifest_sha256={contract['manifest_sha256']}, output={output_dir}",
        flush=True,
    )
    encoder = OrderedEncoder(inventory, start, args.sample_rate, args.workers)
    name: str | None = None
    tar_path: pathlib.Path | None = None
    part_path: pathlib.Path | None = None
    archive: tarfile.TarFile | None = None
    index_rows: list[dict[str, Any]] = []
    payload_bytes = 0
    estimated_bytes = 0
    duration_seconds = 0.0
    quantization = _empty_quantization()
    amplitude_scaling = _empty_amplitude_scaling()
    started_at = time.monotonic()
    next_progress_at = started_at + args.progress_interval_seconds
    processed_since_start = 0
    scaled_examples_logged = 0

    def log_progress(force: bool = False) -> None:
        nonlocal next_progress_at
        now = time.monotonic()
        if not force and now < next_progress_at:
            return
        elapsed = max(now - started_at, EPSILON)
        processed = start + processed_since_start
        rate = processed_since_start / elapsed
        remaining = len(inventory) - processed
        eta = (remaining / rate) if rate > 0.0 else None
        total_duration = prior_duration_seconds + duration_seconds
        total_scaling = _merge_amplitude_scaling(prior_scaling, amplitude_scaling)
        eta_text = f"{eta / 60.0:.1f}m" if eta is not None else "estimating"
        current_gib = estimated_bytes / 1024**3
        print(
            "[packed-shards] progress: "
            f"{processed}/{len(inventory)} ({(100.0 * processed / len(inventory)):.2f}%), "
            f"{rate:.2f} samples/s, elapsed={elapsed / 60.0:.1f}m, eta={eta_text}, "
            f"audio={total_duration / 3600.0:.2f}h, completed_shards={len(state['shards'])}, "
            f"current_shard={current_gib:.3f} GiB, "
            f"scaled={int(total_scaling['scaled_sample_count'])}, "
            f"max_gain={float(total_scaling['max_restore_gain']):.7g}",
            flush=True,
        )
        while next_progress_at <= now:
            next_progress_at += args.progress_interval_seconds

    def finish_current() -> None:
        nonlocal name, tar_path, part_path, archive, index_rows, payload_bytes
        nonlocal estimated_bytes, duration_seconds, quantization, amplitude_scaling
        nonlocal prior_duration_seconds, prior_scaling
        if archive is None or name is None or tar_path is None or part_path is None:
            return
        _finalize_shard(
            output_dir,
            state,
            name,
            tar_path,
            part_path,
            archive,
            index_rows,
            payload_bytes,
            duration_seconds,
            quantization,
            amplitude_scaling,
        )
        tar_gib = (output_dir / SHARDS_DIRNAME / f"{name}.tar").stat().st_size / 1024**3
        print(
            f"[packed-shards] finalized {name}: {len(index_rows)} samples, "
            f"{tar_gib:.3f} GiB TAR, {duration_seconds / 3600.0:.2f} audio hours",
            flush=True,
        )
        prior_duration_seconds += duration_seconds
        prior_scaling = _merge_amplitude_scaling(prior_scaling, amplitude_scaling)
        name = tar_path = part_path = archive = None
        index_rows = []
        payload_bytes = 0
        estimated_bytes = 0
        duration_seconds = 0.0
        quantization = _empty_quantization()
        amplitude_scaling = _empty_amplitude_scaling()

    try:
        for sample in encoder:
            processed_since_start += 1
            proposed_bytes = _estimated_member_bytes(sample)
            if proposed_bytes + TAR_FINAL_RECORD_BYTES > MAX_SHARD_BYTES:
                raise PackingError(
                    f"sample {sample.record.sample_id!r} encodes to a TAR member pair larger than "
                    "the 2 GiB shard cap; split or exclude it explicitly in the authoritative manifest"
                )
            if (
                archive is not None
                and index_rows
                and estimated_bytes + proposed_bytes + TAR_FINAL_RECORD_BYTES > target_bytes
            ):
                finish_current()
            if archive is None:
                name, tar_path, part_path, archive = _start_shard(output_dir, state)
            stem = _member_stem(sample.record.sample_id)
            audio_member = f"samples/{stem}.flac"
            metadata_member = f"samples/{stem}.json"
            _tar_add_bytes(archive, audio_member, sample.audio)
            _tar_add_bytes(archive, metadata_member, sample.metadata)
            index_rows.append(
                {
                    "sample_id": sample.record.sample_id,
                    "shard": f"{SHARDS_DIRNAME}/{name}.tar",
                    "flac_member": audio_member,
                    "json_member": metadata_member,
                    "duration_seconds": sample.duration_seconds,
                    "source_dataset": sample.record.source_dataset,
                    "frame_count": sample.frame_count,
                    "encoded_byte_size": len(sample.audio),
                    "amplitude_restore_gain": sample.amplitude_restore_gain,
                    "canonical_peak": sample.canonical_peak,
                    "storage_peak": sample.storage_peak,
                }
            )
            payload_bytes += len(sample.audio)
            estimated_bytes += proposed_bytes
            duration_seconds += sample.duration_seconds
            quantization = _merge_quantization(quantization, sample.quantization)
            amplitude_scaling = _merge_amplitude_scaling(
                amplitude_scaling,
                _sample_amplitude_scaling(sample.amplitude_restore_gain, sample.canonical_peak),
            )
            if sample.amplitude_restore_gain > 1.0 and scaled_examples_logged < SCALING_EXAMPLE_LIMIT:
                print(
                    "[packed-shards] scaled storage example: "
                    f"id={sample.record.sample_id!r}, path={sample.record.source_path}, "
                    f"canonical_peak={sample.canonical_peak:.7g}, "
                    f"restore_gain={sample.amplitude_restore_gain:.7g}",
                    flush=True,
                )
                scaled_examples_logged += 1
            log_progress()
        finish_current()
    except BaseException:
        if archive is not None:
            archive.close()
        raise

    if int(state["next_record"]) != len(inventory):
        raise PackingError(
            f"packing stopped at {state['next_record']} records but inventory has {len(inventory)}"
        )
    _build_final_index(output_dir, state["shards"])
    descriptor = _build_descriptor(contract, state)
    pending_descriptor = output_dir / f".{DESCRIPTOR_FILENAME}.pending"
    _atomic_write_json(pending_descriptor, descriptor)
    # The final descriptor is the completion marker.  Verify against a private
    # pending descriptor first, then publish it atomically only after success.
    verify_output(output_dir, descriptor_path=pending_descriptor)
    os.replace(pending_descriptor, output_dir / DESCRIPTOR_FILENAME)
    _cleanup_completed_state(output_dir, state)
    print(
        f"[packed-shards] complete: {descriptor['counts']['samples']} samples in "
        f"{descriptor['counts']['shards']} shards, "
        f"{descriptor['counts']['total_duration_seconds'] / 3600.0:.2f} audio hours, "
        f"scaled={descriptor['amplitude_scaling']['scaled_sample_count']}, "
        f"max_gain={descriptor['amplitude_scaling']['max_restore_gain']:.7g}: {output_dir}",
        flush=True,
    )


def _read_index(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PackingError(f"invalid index JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise PackingError(f"index row {line_number} is not an object")
            rows.append(row)
    return rows


def _safe_relative_path(value: Any, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str):
        raise PackingError(f"{label} must be a string")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise PackingError(f"unsafe {label}: {value!r}")
    return path


def _cleanup_completed_state(output_dir: pathlib.Path, state: dict[str, Any]) -> None:
    """Remove only producer-owned recovery files after final descriptor publication."""
    for shard in state["shards"]:
        part = output_dir / shard["index_part"]
        part.unlink(missing_ok=True)
    try:
        (output_dir / PARTS_DIRNAME).rmdir()
    except OSError:
        # A completed descriptor is already verified and atomically public.  An
        # unexpected leftover recovery file should not turn that success into a
        # false failure; it is harmless and is never consumed by the loader.
        return
    (output_dir / STATE_FILENAME).unlink(missing_ok=True)


def verify_output(output_dir: pathlib.Path, descriptor_path: pathlib.Path | None = None) -> None:
    """Structural, lossless-format verification without scanning source datasets."""
    output_dir = output_dir.resolve()
    descriptor_path = descriptor_path or output_dir / DESCRIPTOR_FILENAME
    descriptor = _load_json(descriptor_path)
    if descriptor.get("format_version") != FORMAT_VERSION:
        raise PackingError(f"unsupported shard manifest version in {descriptor_path}")
    shards = descriptor.get("shards")
    counts = descriptor.get("counts")
    if not isinstance(shards, list) or not isinstance(counts, dict):
        raise PackingError("descriptor lacks shards/counts")
    index_name = _safe_relative_path(descriptor.get("index"), "descriptor index")
    index_rows = _read_index(output_dir / index_name)
    if len(index_rows) != counts.get("samples"):
        raise PackingError("descriptor sample count does not match index row count")
    ids: set[str] = set()
    members: set[str] = set()
    referenced_by_shard: dict[str, list[dict[str, Any]]] = {}
    for row in index_rows:
        required = {
            "sample_id",
            "shard",
            "flac_member",
            "json_member",
            "duration_seconds",
            "source_dataset",
            "frame_count",
            "encoded_byte_size",
        }
        missing = required - row.keys()
        if missing:
            raise PackingError(f"index row lacks required fields: {sorted(missing)}")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or sample_id in ids:
            raise PackingError(f"duplicate or invalid sample_id in index: {sample_id!r}")
        ids.add(sample_id)
        shard_path = str(_safe_relative_path(row["shard"], "index shard"))
        for member_label in ("flac_member", "json_member"):
            member = _safe_relative_path(row[member_label], f"index {member_label}")
            if member.parts[0] != "samples":
                raise PackingError(f"unexpected member path: {member}")
            if str(member) in members:
                raise PackingError(f"duplicate member name in index: {member}")
            members.add(str(member))
        expected_stem = _member_stem(sample_id)
        if (
            row["flac_member"] != f"samples/{expected_stem}.flac"
            or row["json_member"] != f"samples/{expected_stem}.json"
        ):
            raise PackingError(f"member key does not match stable sample ID for {sample_id}")
        _optional_restore_gain(row.get("amplitude_restore_gain"), f"index restore gain for {sample_id}")
        row_canonical_peak = _optional_nonnegative_float(
            row.get("canonical_peak"), f"index canonical peak for {sample_id}"
        )
        row_storage_peak = _optional_nonnegative_float(
            row.get("storage_peak"), f"index storage peak for {sample_id}"
        )
        if (row_canonical_peak is None) != (row_storage_peak is None):
            raise PackingError(f"index peak fields must both be present or absent for {sample_id}")
        if pathlib.PurePosixPath(row["flac_member"]).with_suffix("") != pathlib.PurePosixPath(
            row["json_member"]
        ).with_suffix(""):
            raise PackingError(f"audio/metadata member pair does not share a stem for {sample_id}")
        referenced_by_shard.setdefault(shard_path, []).append(row)
    if len(shards) != counts.get("shards"):
        raise PackingError("descriptor shard count does not match shard list")

    descriptor_paths: set[str] = set()
    total_duration = 0.0
    total_tar_bytes = 0
    total_amplitude_scaling = _empty_amplitude_scaling()
    for shard in shards:
        if not isinstance(shard, dict):
            raise PackingError("descriptor shard entry is not an object")
        relative = str(_safe_relative_path(shard.get("path"), "descriptor shard path"))
        if relative in descriptor_paths:
            raise PackingError(f"duplicate shard path in descriptor: {relative}")
        descriptor_paths.add(relative)
        rows = referenced_by_shard.get(relative, [])
        if len(rows) != shard.get("count"):
            raise PackingError(f"index count mismatch for {relative}")
        shard_scaling = _shard_amplitude_scaling(shard)
        if int(shard_scaling["scaled_sample_count"]) > int(shard["count"]):
            raise PackingError(f"scaled sample count exceeds shard sample count for {relative}")
        tar_path = output_dir / relative
        if not tar_path.is_file():
            raise PackingError(f"missing shard: {tar_path}")
        if tar_path.stat().st_size != shard.get("tar_bytes"):
            raise PackingError(f"TAR byte count mismatch for {relative}")
        # ``r:`` intentionally refuses gzip/bzip/xz wrappers; TAR must remain
        # uncompressed because FLAC members are compressed individually.
        try:
            with tarfile.open(tar_path, mode="r:") as archive:
                members = archive.getmembers()
                names = [member.name for member in members]
                if len(names) != len(set(names)):
                    raise PackingError(f"duplicate TAR member names in {relative}")
                expected = {row["flac_member"] for row in rows} | {row["json_member"] for row in rows}
                if set(names) != expected:
                    raise PackingError(f"TAR members do not match index for {relative}")
                observed_scaling = _empty_amplitude_scaling()
                for row in rows:
                    flac = archive.getmember(row["flac_member"])
                    metadata = archive.getmember(row["json_member"])
                    if not flac.isfile() or not metadata.isfile():
                        raise PackingError(f"non-regular member in {relative} for {row['sample_id']}")
                    if flac.size != row["encoded_byte_size"]:
                        raise PackingError(f"encoded-byte mismatch for {row['sample_id']}")
                    flac_file = archive.extractfile(flac)
                    metadata_file = archive.extractfile(metadata)
                    if flac_file is None or metadata_file is None:
                        raise PackingError(f"cannot read member for {row['sample_id']}")
                    flac_bytes = flac_file.read()
                    info = sf.info(io.BytesIO(flac_bytes))
                    if int(info.samplerate) != int(descriptor["sample_rate"]) or int(info.channels) != 1:
                        raise PackingError(f"unexpected FLAC audio shape for {row['sample_id']}")
                    if str(info.subtype) != "PCM_16":
                        raise PackingError(f"unexpected FLAC subtype for {row['sample_id']}: {info.subtype}")
                    if int(info.frames) != int(row["frame_count"]):
                        raise PackingError(f"frame-count mismatch for {row['sample_id']}")
                    expected_duration = int(row["frame_count"]) / int(descriptor["sample_rate"])
                    if not math.isclose(float(row["duration_seconds"]), expected_duration, abs_tol=1e-12):
                        raise PackingError(f"duration mismatch for {row['sample_id']}")
                    try:
                        packed_meta = json.loads(metadata_file.read().decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise PackingError(f"invalid adjacent JSON for {row['sample_id']}: {exc}") from exc
                    packed = packed_meta.get("packed") if isinstance(packed_meta, dict) else None
                    if not isinstance(packed, dict) or packed.get("sample_id") != row["sample_id"]:
                        raise PackingError(f"metadata identity mismatch for {row['sample_id']}")
                    if (
                        packed.get("source_dataset") != row["source_dataset"]
                        or packed.get("canonical_sample_rate") != descriptor["sample_rate"]
                        or packed.get("canonical_channels") != 1
                        or packed.get("canonical_subtype") != "PCM_16"
                        or packed.get("canonical_frame_count") != row["frame_count"]
                    ):
                        raise PackingError(f"metadata canonical fields mismatch for {row['sample_id']}")
                    if not isinstance(packed_meta.get("original"), dict):
                        raise PackingError(f"metadata original row missing for {row['sample_id']}")
                    row_gain = _optional_restore_gain(
                        row.get("amplitude_restore_gain"), f"index restore gain for {row['sample_id']}"
                    )
                    packed_gain = _optional_restore_gain(
                        packed.get("amplitude_restore_gain"),
                        f"metadata restore gain for {row['sample_id']}",
                    )
                    if not math.isclose(row_gain, packed_gain, rel_tol=0.0, abs_tol=1e-12):
                        raise PackingError(f"restore gain mismatch for {row['sample_id']}")
                    row_canonical_peak = _optional_nonnegative_float(
                        row.get("canonical_peak"), f"index canonical peak for {row['sample_id']}"
                    )
                    packed_canonical_peak = _optional_nonnegative_float(
                        packed.get("canonical_peak"),
                        f"metadata canonical peak for {row['sample_id']}",
                    )
                    row_storage_peak = _optional_nonnegative_float(
                        row.get("storage_peak"), f"index storage peak for {row['sample_id']}"
                    )
                    packed_storage_peak = _optional_nonnegative_float(
                        packed.get("storage_peak"),
                        f"metadata storage peak for {row['sample_id']}",
                    )
                    if (packed_canonical_peak is None) != (packed_storage_peak is None):
                        raise PackingError(
                            f"metadata peak fields must both be present or absent for {row['sample_id']}"
                        )
                    if (row_canonical_peak is None) != (packed_canonical_peak is None) or (
                        row_storage_peak is None
                    ) != (packed_storage_peak is None):
                        raise PackingError(f"peak metadata presence mismatch for {row['sample_id']}")
                    if row_canonical_peak is not None and (
                        not math.isclose(
                            row_canonical_peak, packed_canonical_peak, rel_tol=0.0, abs_tol=1e-12
                        )
                        or not math.isclose(
                            row_storage_peak, packed_storage_peak, rel_tol=0.0, abs_tol=1e-12
                        )
                        or packed_storage_peak > PCM16_STORAGE_PEAK
                    ):
                        raise PackingError(f"peak metadata mismatch or unsafe storage peak for {row['sample_id']}")
                    observed_scaling = _merge_amplitude_scaling(
                        observed_scaling,
                        _sample_amplitude_scaling(
                            packed_gain,
                            packed_canonical_peak if packed_canonical_peak is not None else 0.0,
                        ),
                    )
        except tarfile.TarError as exc:
            raise PackingError(f"invalid or compressed TAR {relative}: {exc}") from exc
        if not _same_amplitude_scaling(shard_scaling, observed_scaling):
            raise PackingError(f"declared amplitude scaling does not match samples in {relative}")
        total_amplitude_scaling = _merge_amplitude_scaling(
            total_amplitude_scaling, observed_scaling
        )
        total_duration += float(shard["duration_seconds"])
        total_tar_bytes += int(shard["tar_bytes"])
    if set(referenced_by_shard) != descriptor_paths:
        raise PackingError("index references a shard absent from descriptor")
    if not math.isclose(total_duration, float(counts["total_duration_seconds"]), abs_tol=1e-9):
        raise PackingError("descriptor total duration does not match shards")
    if total_tar_bytes != int(counts["total_tar_bytes"]):
        raise PackingError("descriptor total TAR bytes do not match shards")
    descriptor_scaling = descriptor.get("amplitude_scaling")
    if descriptor_scaling is not None:
        declared_scaling = _shard_amplitude_scaling({"amplitude_scaling": descriptor_scaling})
        if not _same_amplitude_scaling(declared_scaling, total_amplitude_scaling):
            raise PackingError("descriptor amplitude scaling does not match shards")
    print(f"[packed-shards] verification passed: {len(index_rows)} samples, {len(shards)} shards")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack_parser = subparsers.add_parser("pack", help="canonicalize manifest rows directly into TAR shards")
    pack_parser.add_argument("--manifest", required=True, help="authoritative combined training JSONL")
    pack_parser.add_argument("--output-dir", required=True, help="new packed output directory")
    pack_parser.add_argument("--sample-rate", type=int, default=16000)
    pack_parser.add_argument("--target-shard-size-gb", type=float, default=1.0)
    pack_parser.add_argument("--workers", type=int, default=4, help="bounded CPU decode/encode workers")
    pack_parser.add_argument("--seed", type=int, default=42, help="deterministic manifest shuffle seed")
    pack_parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=30.0,
        help="emit elapsed/rate/ETA packing progress at this positive interval (default: 30)",
    )
    pack_parser.add_argument(
        "--resume",
        action="store_true",
        help="start in an empty output directory or resume only a matching interrupted pack",
    )
    verify_parser = subparsers.add_parser("verify", help="verify an existing packed output without sources")
    verify_parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "pack":
            pack(args)
        elif args.command == "verify":
            verify_output(pathlib.Path(args.output_dir))
        else:  # argparse enforces this; keep the branch explicit for type checkers.
            raise AssertionError(f"unknown command {args.command}")
    except (PackingError, OSError, ValueError) as exc:
        raise SystemExit(f"[packed-shards] ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
