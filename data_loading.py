"""Dataset loading and waveform augmentation."""
from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import pathlib
import random
import tarfile
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import soundfile as sf
import torchaudio
import torch
import torch.nn.functional as F

from schema import SpanMaskCfg, WaveAugCfg


@dataclass
class DatasetConfig:
    manifest: str | list[str]
    sample_rate: int = 16000
    segment_seconds: float = 2.0
    random_crop: bool = True


class PackedShardError(RuntimeError):
    """A packed-shard descriptor or TAR violated the producer contract."""


_PACKED_KIND = "continuous-latent-ae-audio-tar-shards"
_PACKED_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PackedShard:
    """A validated shard descriptor entry, without reading index.jsonl."""

    path: pathlib.Path
    relative_path: str
    count: int


def _packed_safe_relative_path(value: Any, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str):
        raise PackedShardError(f"{label} must be a string")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise PackedShardError(f"unsafe {label}: {value!r}")
    return path


def load_packed_shard_manifest(
    manifest_path: str,
    *,
    expected_sample_rate: int,
) -> list[PackedShard]:
    """Read only the small public descriptor; training never opens index.jsonl."""
    descriptor_path = pathlib.Path(manifest_path).resolve()
    if not descriptor_path.is_file():
        raise PackedShardError(f"shard manifest does not exist: {descriptor_path}")
    try:
        value = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackedShardError(f"cannot read shard manifest {descriptor_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackedShardError("shard manifest must be a JSON object")
    if value.get("kind") != _PACKED_KIND:
        raise PackedShardError(f"unsupported shard manifest kind: {value.get('kind')!r}")
    if value.get("format_version") != _PACKED_FORMAT_VERSION:
        raise PackedShardError(
            f"unsupported shard manifest version: {value.get('format_version')!r}"
        )
    descriptor_rate = value.get("sample_rate")
    if (
        not isinstance(descriptor_rate, int)
        or isinstance(descriptor_rate, bool)
        or descriptor_rate != expected_sample_rate
    ):
        raise PackedShardError(
            f"shard sample rate {descriptor_rate!r} does not match "
            f"data.sample_rate={expected_sample_rate}"
        )
    entries = value.get("shards")
    counts = value.get("counts")
    if not isinstance(entries, list) or not entries:
        raise PackedShardError("shard manifest has no shards")
    total_samples = counts.get("samples") if isinstance(counts, dict) else None
    if (
        not isinstance(total_samples, int)
        or isinstance(total_samples, bool)
        or total_samples <= 0
    ):
        raise PackedShardError("shard manifest has no positive sample count")
    root = descriptor_path.parent.resolve()
    seen: set[str] = set()
    shards: list[PackedShard] = []
    total = 0
    for number, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PackedShardError(f"shard entry {number} is not an object")
        relative = _packed_safe_relative_path(entry.get("path"), f"shard {number} path")
        relative_text = str(relative)
        if relative_text in seen:
            raise PackedShardError(f"duplicate shard path: {relative_text}")
        seen.add(relative_text)
        if relative.suffix != ".tar":
            raise PackedShardError(f"shard {number} is not a .tar file: {relative_text}")
        count = entry.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise PackedShardError(f"shard {number} has invalid positive count: {count!r}")
        candidate = (root / pathlib.Path(*relative.parts)).resolve()
        if candidate != root and root not in candidate.parents:
            raise PackedShardError(f"shard path escapes descriptor directory: {relative_text}")
        if not candidate.is_file():
            raise PackedShardError(f"shard file does not exist: {candidate}")
        shards.append(PackedShard(candidate, relative_text, count))
        total += count
    if total != total_samples:
        raise PackedShardError(
            f"descriptor shard counts total {total}, expected {total_samples} samples"
        )
    return shards


def packed_epoch_assignment(
    shards: Sequence[PackedShard],
    *,
    seed: int,
    epoch: int,
    total_consumers: int,
    batch_size: int,
) -> tuple[list[list[PackedShard]], int]:
    """Deterministically shuffle and greedily balance whole shards by count.

    The common quota is per global consumer, intentionally rounded to a full
    DataLoader batch.  This makes every rank exhaust an epoch together even
    though whole-shard balancing cannot be exact.
    """
    if total_consumers <= 0:
        raise PackedShardError("total consumer count must be positive")
    if batch_size <= 0:
        raise PackedShardError("batch size must be positive")
    if len(shards) < total_consumers:
        raise PackedShardError(
            f"packed training needs at least one shard per global consumer; found "
            f"{len(shards)} shards for {total_consumers} consumers"
        )
    ordered = list(shards)
    # Integer mixing avoids Python's salted hash and is stable across spawned workers.
    rng = random.Random((int(seed) & ((1 << 63) - 1)) ^ (int(epoch) * 0x9E3779B1))
    rng.shuffle(ordered)
    groups: list[list[PackedShard]] = [[] for _ in range(total_consumers)]
    loads = [0] * total_consumers
    for shard in ordered:
        consumer = min(range(total_consumers), key=lambda index: (loads[index], index))
        groups[consumer].append(shard)
        loads[consumer] += shard.count
    quota = min(loads) // batch_size * batch_size
    if quota < batch_size:
        raise PackedShardError(
            f"packed shard assignment yields only {min(loads)} samples for its smallest "
            f"consumer, which cannot form one batch of {batch_size}"
        )
    paths = [shard.relative_path for group in groups for shard in group]
    if len(paths) != len(set(paths)):
        raise AssertionError("packed epoch assignment duplicated a shard")
    return groups, quota


def packed_worker_init(_: int) -> None:
    """Avoid multiplying decode/resample-style CPU pools across workers."""
    torch.set_num_threads(1)


def _packed_buffer_requires_eviction(
    *, item_count: int, buffered_bytes: int, byte_budget: int
) -> bool:
    """Keep the byte budget, except for one unavoidable oversized member."""
    return item_count > 1 and buffered_bytes > byte_budget


def packed_metadata_restore_gain(packed: dict[str, Any]) -> float:
    """Validate optional v1 storage scaling and return its training-time inverse.

    Finalized shards produced before reversible scaling have none of these
    fields and therefore retain their original PCM16 behavior with gain 1.
    New metadata is all-or-nothing so a partial/corrupt wrapper cannot silently
    alter waveform amplitude.
    """
    field_names = ("amplitude_restore_gain", "canonical_peak", "storage_peak")
    present = {name: name in packed for name in field_names}
    if not any(present.values()):
        return 1.0
    if not all(present.values()):
        raise PackedShardError("packed amplitude metadata must be present together")
    gain = packed["amplitude_restore_gain"]
    canonical_peak = packed["canonical_peak"]
    storage_peak = packed["storage_peak"]
    numeric = (gain, canonical_peak, storage_peak)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric):
        raise PackedShardError("packed amplitude metadata must be numeric, not boolean")
    gain = float(gain)
    canonical_peak = float(canonical_peak)
    storage_peak = float(storage_peak)
    if not all(math.isfinite(value) for value in (gain, canonical_peak, storage_peak)):
        raise PackedShardError("packed amplitude metadata must be finite")
    if gain < 1.0:
        raise PackedShardError("packed amplitude_restore_gain must be at least 1")
    if canonical_peak < 0.0 or storage_peak < 0.0:
        raise PackedShardError("packed peak metadata must be non-negative")
    # PCM16 samples are normalized to at most full scale. The producer stores
    # explicit headroom below it, but the loader intentionally does not require
    # that producer-internal threshold to preserve the public v1 contract.
    if storage_peak > 1.0:
        raise PackedShardError("packed storage_peak exceeds PCM16 full scale")
    if canonical_peak < storage_peak:
        raise PackedShardError("packed canonical_peak cannot be below storage_peak")
    if gain > 1.0 and canonical_peak <= storage_peak:
        raise PackedShardError("scaled packed metadata has inconsistent peaks")
    return gain


class PackedTarDataset(torch.utils.data.IterableDataset[dict[str, Any]]):
    """Streaming, duplicate-free TAR dataset for canonical PCM16 FLAC shards."""

    def __init__(
        self,
        *,
        shard_manifest: str,
        sample_rate: int,
        segment_seconds: float,
        random_crop: bool,
        shuffle_buffer_mb: int,
        run_seed: int,
        rank: int,
        world_size: int,
        workers_per_rank: int,
        batch_size: int,
    ):
        super().__init__()
        if world_size <= 0 or not 0 <= rank < world_size:
            raise PackedShardError(f"invalid DDP rank/world_size: {rank}/{world_size}")
        if workers_per_rank < 0:
            raise PackedShardError("workers_per_rank cannot be negative")
        if shuffle_buffer_mb <= 0:
            raise PackedShardError("shuffle_buffer_mb must be positive")
        self.shards = load_packed_shard_manifest(
            shard_manifest, expected_sample_rate=sample_rate
        )
        self.sample_rate = sample_rate
        self.num_samples = int(math.ceil(segment_seconds * sample_rate))
        self.random_crop = random_crop
        self.max_buffer_bytes = int(shuffle_buffer_mb) * 1024 * 1024
        self.run_seed = int(run_seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.workers_per_rank = max(1, int(workers_per_rank))
        self.batch_size = int(batch_size)
        # multiprocessing.Value is a spawn-pickleable shared primitive; workers
        # retain it with persistent_workers and observe set_epoch without loader
        # recreation.
        # The DataLoader explicitly uses ``spawn``. Creating this SemLock from
        # that same context makes the synchronized value pickleable into spawned
        # workers on Linux as well as Windows/macOS; a default/fork-context Value
        # can otherwise be rejected during worker process start.
        self._epoch = mp.get_context("spawn").Value("q", 0, lock=True)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("data epoch cannot be negative")
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _epoch_value(self) -> int:
        with self._epoch.get_lock():
            return int(self._epoch.value)

    @staticmethod
    def _safe_member_name(name: str, suffix: str) -> str:
        path = pathlib.PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 2
            or path.parts[0] != "samples"
            or not path.name.endswith(suffix)
        ):
            raise PackedShardError(f"unsafe or unexpected TAR member name: {name!r}")
        stem = path.name[: -len(suffix)]
        if len(stem) != 64 or any(character not in "0123456789abcdef" for character in stem):
            raise PackedShardError(f"unexpected TAR member key: {name!r}")
        return stem

    @classmethod
    def _read_pair(
        cls, archive: tarfile.TarFile, first: tarfile.TarInfo, *, expected_sample_rate: int
    ) -> tuple[bytes, dict[str, Any]]:
        if not first.isfile():
            raise PackedShardError(f"TAR member is not a regular file: {first.name!r}")
        stem = cls._safe_member_name(first.name, ".flac")
        audio_file = archive.extractfile(first)
        if audio_file is None:
            raise PackedShardError(f"cannot read TAR member: {first.name!r}")
        audio = audio_file.read()
        second = archive.next()
        if second is None or not second.isfile():
            raise PackedShardError(f"missing adjacent JSON member after {first.name!r}")
        json_stem = cls._safe_member_name(second.name, ".json")
        if json_stem != stem:
            raise PackedShardError(f"TAR FLAC/JSON pair mismatch: {first.name!r}, {second.name!r}")
        metadata_file = archive.extractfile(second)
        if metadata_file is None:
            raise PackedShardError(f"cannot read TAR member: {second.name!r}")
        try:
            wrapper = json.loads(metadata_file.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackedShardError(f"invalid metadata JSON for {second.name!r}: {exc}") from exc
        if not isinstance(wrapper, dict) or not isinstance(wrapper.get("original"), dict):
            raise PackedShardError(f"invalid metadata wrapper for {second.name!r}")
        original = wrapper["original"]
        packed = wrapper.get("packed")
        if not isinstance(packed, dict) or not isinstance(packed.get("sample_id"), str):
            raise PackedShardError(f"metadata wrapper lacks packed sample ID for {second.name!r}")
        source_dataset = str(original.get("dataset") or "unknown")
        original_id = original.get("id")
        if original_id is not None and str(original_id) != "":
            expected_sample_id = f"{source_dataset}:{original_id}"
        else:
            try:
                original_bytes = json.dumps(
                    original,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise PackedShardError(f"invalid original metadata for {second.name!r}: {exc}") from exc
            expected_sample_id = f"{source_dataset}:sha256:{hashlib.sha256(original_bytes).hexdigest()}"
        if (
            packed["sample_id"] != expected_sample_id
            or packed.get("source_dataset") != source_dataset
        ):
            raise PackedShardError(f"wrapped metadata identity mismatch for {second.name!r}")
        expected_stem = hashlib.sha256(expected_sample_id.encode("utf-8")).hexdigest()
        if stem != expected_stem:
            raise PackedShardError(f"metadata sample ID does not match TAR member key: {second.name!r}")
        if (
            packed.get("canonical_sample_rate") != expected_sample_rate
            or packed.get("canonical_channels") != 1
            or packed.get("canonical_subtype") != "PCM_16"
            or not isinstance(packed.get("canonical_frame_count"), int)
            or isinstance(packed.get("canonical_frame_count"), bool)
            or packed["canonical_frame_count"] <= 0
        ):
            raise PackedShardError(f"invalid canonical metadata for {second.name!r}")
        packed_metadata_restore_gain(packed)
        return audio, wrapper

    def _decode_and_crop(
        self, audio: bytes, wrapper: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        try:
            wav, rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
            info = sf.info(io.BytesIO(audio))
        except Exception as exc:
            raise PackedShardError(f"failed to decode packed FLAC: {exc}") from exc
        packed = wrapper["packed"]
        if (
            int(rate) != self.sample_rate
            or int(info.samplerate) != self.sample_rate
            or int(info.channels) != 1
            or str(info.subtype) != "PCM_16"
            or wav.ndim != 2
            or wav.shape[1] != 1
            or wav.shape[0] != packed["canonical_frame_count"]
        ):
            raise PackedShardError(
                "packed FLAC violates canonical rate/channel/frame contract for "
                f"{packed.get('sample_id')!r}"
            )
        # This reverses storage-only PCM16 scaling before the exact existing
        # crop/pad path. It is deliberately not a clamp or normalization.
        wav = wav * packed_metadata_restore_gain(packed)
        tensor = torch.from_numpy(wav[:, 0].copy())
        if self.random_crop:
            if tensor.numel() < self.num_samples:
                tensor = F.pad(tensor, (0, self.num_samples - tensor.numel()))
            else:
                start = rng.randint(0, tensor.numel() - self.num_samples)
                tensor = tensor[start : start + self.num_samples]
        else:
            tensor = _start_crop(tensor, self.num_samples)
        return {"wav": tensor, "meta": wrapper["original"]}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        actual_workers = worker.num_workers if worker is not None else 1
        if actual_workers != self.workers_per_rank:
            raise PackedShardError(
                f"packed dataset was configured for {self.workers_per_rank} workers but "
                f"DataLoader started {actual_workers}; rebuild it with matching data.num_workers"
            )
        epoch = self._epoch_value()
        total_consumers = self.world_size * self.workers_per_rank
        groups, quota = packed_epoch_assignment(
            self.shards,
            seed=self.run_seed,
            epoch=epoch,
            total_consumers=total_consumers,
            batch_size=self.batch_size,
        )
        consumer_id = self.rank * self.workers_per_rank + worker_id
        assigned = groups[consumer_id]
        # The RNG serves only this consumer and epoch. It controls uniform quota
        # selection, buffer eviction/drain, and crop locations without depending
        # on DataLoader's global worker seed.
        rng = random.Random(
            ((self.run_seed & ((1 << 63) - 1)) << 17)
            ^ (epoch * 0x85EBCA6B)
            ^ (consumer_id * 0xC2B2AE35)
        )
        remaining_total = sum(shard.count for shard in assigned)
        remaining_select = quota
        buffer: list[tuple[bytes, dict[str, Any], int]] = []
        buffered_bytes = 0
        yielded = 0
        for shard in assigned:
            try:
                archive = tarfile.open(shard.path, mode="r|")
            except (OSError, tarfile.TarError) as exc:
                raise PackedShardError(f"cannot open packed shard {shard.path}: {exc}") from exc
            shard_seen = 0
            with archive:
                while True:
                    member = archive.next()
                    if member is None:
                        break
                    audio, wrapper = self._read_pair(
                        archive, member, expected_sample_rate=self.sample_rate
                    )
                    shard_seen += 1
                    if remaining_total <= 0:
                        raise PackedShardError("descriptor count underflow while streaming shards")
                    choose = remaining_select > 0 and rng.randrange(remaining_total) < remaining_select
                    remaining_total -= 1
                    if not choose:
                        continue
                    remaining_select -= 1
                    cost = len(audio) + len(json.dumps(wrapper, ensure_ascii=False).encode("utf-8"))
                    buffer.append((audio, wrapper, cost))
                    buffered_bytes += cost
                    # A single member can legitimately exceed the nominal byte
                    # budget. Keep at most that one member above the budget; this
                    # is bounded by the largest selected FLAC+JSON pair rather
                    # than silently rejecting a valid producer shard.
                    while _packed_buffer_requires_eviction(
                        item_count=len(buffer),
                        buffered_bytes=buffered_bytes,
                        byte_budget=self.max_buffer_bytes,
                    ):
                        index = rng.randrange(len(buffer))
                        selected_audio, selected_wrapper, selected_cost = buffer.pop(index)
                        buffered_bytes -= selected_cost
                        yield self._decode_and_crop(selected_audio, selected_wrapper, rng)
                        yielded += 1
            if shard_seen != shard.count:
                raise PackedShardError(
                    f"shard {shard.relative_path} contains {shard_seen} samples, descriptor says {shard.count}"
                )
        if remaining_total != 0 or remaining_select != 0:
            raise PackedShardError("packed shard sample counts did not match descriptor assignment")
        while buffer:
            index = rng.randrange(len(buffer))
            audio, wrapper, cost = buffer.pop(index)
            buffered_bytes -= cost
            yield self._decode_and_crop(audio, wrapper, rng)
            yielded += 1
        if yielded != quota:
            raise AssertionError(f"packed consumer yielded {yielded}, expected quota {quota}")


def _read_manifest(paths: str | list[str]) -> list[tuple[dict[str, Any], pathlib.Path]]:
    if isinstance(paths, str):
        paths = [paths]
    records: list[tuple[dict[str, Any], pathlib.Path]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            items = [json.loads(line) for line in file if line.strip()]
        root = resolve_manifest_root(path, items)
        records.extend((item, root) for item in items)
    return records


def _random_crop(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    if wav.numel() < num_samples:
        return F.pad(wav, (0, num_samples - wav.numel()))
    start = random.randint(0, wav.numel() - num_samples)
    return wav[start : start + num_samples]


def _start_crop(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    if wav.numel() < num_samples:
        return F.pad(wav, (0, num_samples - wav.numel()))
    return wav[:num_samples]


def resolve_manifest_root(
    manifest_path: str, items: list[dict[str, Any]]
) -> pathlib.Path:
    """Directory against which relative ``audio_filepath`` rows resolve.

    Two layouts exist in the wild: manifests sitting next to the audio
    (root = the manifest's own directory), and the packed (`scripts/housekeeping.py`) layout
    ``<root>/manifests/*.jsonl`` whose rows are relative to ``<root>``
    (e.g. ``audio/openslr53/utt1.flac`` — root = one level up). Probe the
    first relative row against both candidates so a wrong guess fails here,
    at construction, rather than mid-epoch inside a dataloader worker.
    """
    parent = pathlib.Path(manifest_path).resolve().parent
    for it in items:
        p = it.get("audio_filepath")
        if not p or os.path.isabs(p):
            continue
        for cand in (parent, parent.parent):
            if (cand / p).exists():
                return cand
        raise FileNotFoundError(
            f"relative audio_filepath {p!r} from {manifest_path} not found "
            f"under {parent} or {parent.parent}"
        )
    return parent


class AudioDataset(torch.utils.data.Dataset):
    """Map-style dataset over one or more JSONL manifests."""

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        records = _read_manifest(cfg.manifest)
        self.items = [item for item, _ in records]
        self._manifest_roots = [root for _, root in records]
        self.num_samples = int(math.ceil(cfg.segment_seconds * cfg.sample_rate))
        self._resamplers: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _resample(self, wav: torch.Tensor, src_sr: int) -> torch.Tensor:
        if src_sr == self.cfg.sample_rate:
            return wav
        if src_sr not in self._resamplers:
            self._resamplers[src_sr] = torchaudio.transforms.Resample(
                src_sr, self.cfg.sample_rate
            )
        return self._resamplers[src_sr](wav)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        path = item["audio_filepath"]
        if not os.path.isabs(path):
            path = str(self._manifest_roots[idx] / path)
        wav, sr = torchaudio.load(path)
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.flatten()
        wav = self._resample(wav, int(sr))
        if self.cfg.random_crop:
            wav = _random_crop(wav, self.num_samples)
        else:
            wav = _start_crop(wav, self.num_samples)
        return {"wav": wav, "meta": item}


def collate_fixed(batch: list[dict[str, Any]]) -> dict[str, Any]:
    wav = torch.stack([item["wav"] for item in batch], dim=0).unsqueeze(1)
    meta = [item["meta"] for item in batch]
    return {"wav": wav, "meta": meta}


def make_span_masks(
    batch_size: int,
    num_frames: int,
    cfg: SpanMaskCfg,
) -> torch.Tensor:
    """Build CPU masks of contiguous frame spans."""
    masks = torch.zeros((batch_size, num_frames), dtype=torch.float32)
    if not cfg.enabled or num_frames <= 0 or cfg.ratio == 0.0:
        return masks
    target = max(1, int(round(num_frames * cfg.ratio)))
    min_span = min(cfg.min_span_frames, num_frames)
    max_span = min(cfg.max_span_frames, num_frames)
    for b in range(batch_size):
        covered = 0
        attempts = 0
        max_attempts = max(50, target * 4)
        while covered < target and attempts < max_attempts:
            span = random.randint(min_span, max_span)
            span = min(span, num_frames)
            start = random.randint(0, num_frames - span)
            before = int(masks[b, start:start + span].sum().item())
            masks[b, start:start + span] = 1.0
            covered += span - before
            attempts += 1
    return masks


def apply_waveform_chunk_mask(
    wav: torch.Tensor,
    frame_masks: torch.Tensor,
    samples_per_frame: int,
) -> torch.Tensor:
    """Expand frontend-frame masks and apply them to waveform samples."""
    waveform_length = wav.shape[-1]
    sample_mask = frame_masks.repeat_interleave(int(samples_per_frame), dim=-1)
    if sample_mask.size(1) > waveform_length:
        sample_mask = sample_mask[:, :waveform_length]
    elif sample_mask.size(1) < waveform_length:
        sample_mask = F.pad(sample_mask, (0, waveform_length - sample_mask.size(1)))
    sample_mask = sample_mask.to(device=wav.device, dtype=wav.dtype)
    return wav * (1.0 - sample_mask.unsqueeze(1))


def apply_frame_mask(x: torch.Tensor, frame_masks: torch.Tensor) -> torch.Tensor:
    """Zero complete frames in a channels-first feature tensor."""
    mask = frame_masks.to(device=x.device, dtype=x.dtype).unsqueeze(1)
    return x * (1.0 - mask)


def apply_waveform_augment(wav: torch.Tensor, sample_rate: int, cfg: WaveAugCfg) -> torch.Tensor:
    """Apply independent waveform augmentations to each batch item."""
    if not cfg.enabled:
        return wav

    batch_size = wav.shape[0]
    device = wav.device
    dtype = wav.dtype

    if cfg.gain_prob > 0:
        gain_mask = (torch.rand(batch_size, 1, 1, device=device) < cfg.gain_prob).to(dtype)
        gain = torch.empty(batch_size, 1, 1, device=device, dtype=dtype).uniform_(
            cfg.gain_min, cfg.gain_max
        )
        wav = wav * (gain_mask * gain + (1.0 - gain_mask))

    if cfg.noise_prob > 0:
        noise_mask = (torch.rand(batch_size, 1, 1, device=device) < cfg.noise_prob).to(dtype)
        snr_db = torch.empty(batch_size, 1, 1, device=device, dtype=dtype).uniform_(
            cfg.noise_snr_min, cfg.noise_snr_max
        )
        noise = torch.randn_like(wav)
        ra = wav.pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        rb = noise.pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        target_ratio = 10.0 ** (snr_db / 20.0)
        scale = (ra / (rb * target_ratio)).clamp_min(0.0)
        wav = wav + noise_mask * (noise * scale)

    if cfg.lowpass_prob > 0:
        taps = 63
        cutoff = torch.empty(batch_size, 1, device=device).uniform_(
            cfg.lowpass_min_freq, cfg.lowpass_max_freq
        )
        fc = (cutoff / float(sample_rate)).clamp(max=0.5)
        t = torch.arange(taps, device=device, dtype=torch.float32) - (taps - 1) / 2
        h = 2.0 * fc * torch.sinc(2.0 * fc * t)
        h = h * torch.hann_window(taps, periodic=False, device=device)
        h = (h / h.sum(dim=-1, keepdim=True)).to(dtype)
        apply_mask = torch.rand(batch_size, 1, 1, device=device) < cfg.lowpass_prob
        xp = F.pad(wav, (taps // 2, taps // 2), mode="reflect")
        filtered = F.conv1d(
            xp.transpose(0, 1), h.unsqueeze(1), groups=batch_size
        ).transpose(0, 1).to(dtype)
        wav = torch.where(apply_mask, filtered, wav)

    if cfg.clip_prob > 0:
        clip_mask = torch.rand(batch_size, 1, 1, device=device) < cfg.clip_prob
        peak = wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        thresh = torch.empty(batch_size, 1, 1, device=device, dtype=dtype).uniform_(
            cfg.clip_min, 0.99
        ) * peak
        clipped = torch.maximum(torch.minimum(wav, thresh), -thresh)
        wav = torch.where(clip_mask, clipped, wav)

    return wav
