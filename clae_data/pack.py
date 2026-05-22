"""Resample + transcode + split + manifest emission for a packed dataset dir."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from clae_data.adapters.base import DatasetAdapter
from clae_data.audit import audit_records
from clae_data.schema import Record


def _safe_id(rec: Record) -> str:
    """Stable filename stem: prefer ``id``, fall back to a hash of the path."""
    rid = rec.get("id")
    if rid:
        # Sanitize: replace path separators / spaces to keep the filename safe.
        return str(rid).replace("/", "_").replace("\\", "_").replace(" ", "_")
    src = rec.get("audio_filepath", "")
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]


def _transcode_one(
    rec: Record,
    staging_dir: Path,
    target_sr: int,
    skip_existing: bool,
) -> Record | None:
    """Convert one source file to ``staging_dir/audio/<dataset>/<id>.flac``.

    Returns the rewritten Record (with relative audio_filepath, updated
    sample_rate, recomputed duration) on success; ``None`` on failure.
    """
    import soundfile as sf
    import torchaudio
    import torchaudio.functional as AF

    dataset = rec["dataset"]
    stem = _safe_id(rec)
    rel_path = Path("audio") / dataset / f"{stem}.flac"
    out_path = staging_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_rec: Record = dict(rec)  # shallow copy; we don't mutate caller's record

    if skip_existing and out_path.exists():
        # Trust prior transcode. Refresh duration from the existing file so
        # the manifest is internally consistent.
        try:
            info = sf.info(str(out_path))
            new_rec["duration"] = float(info.duration)
            new_rec["sample_rate"] = int(info.samplerate)
        except Exception:
            # Fall through to a fresh transcode if the cached file is bad.
            out_path.unlink(missing_ok=True)
        else:
            new_rec["audio_filepath"] = str(rel_path)
            return new_rec

    try:
        wav, sr = torchaudio.load(rec["audio_filepath"])
    except Exception as e:
        print(f"[pack] decode failed for {rec.get('audio_filepath')}: {e}")
        return None

    try:
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != int(target_sr):
            wav = AF.resample(wav, int(sr), int(target_sr))
        samples = wav.squeeze(0).contiguous().cpu().numpy()
        sf.write(
            str(out_path),
            samples,
            int(target_sr),
            format="FLAC",
            subtype="PCM_16",
        )
    except Exception as e:
        print(f"[pack] transcode failed for {rec.get('audio_filepath')}: {e}")
        return None

    new_rec["audio_filepath"] = str(rel_path)
    new_rec["sample_rate"] = int(target_sr)
    new_rec["duration"] = float(samples.shape[-1] / target_sr)
    return new_rec


def _per_dataset_split(
    records: list[Record], val_pct: float, rng: random.Random
) -> tuple[list[Record], list[Record]]:
    """Split per ``dataset`` so every source contributes to both splits."""
    by_ds: dict[str, list[Record]] = {}
    for r in records:
        by_ds.setdefault(r["dataset"], []).append(r)

    train: list[Record] = []
    val: list[Record] = []
    for ds, rows in by_ds.items():
        rng.shuffle(rows)
        n = len(rows)
        if n == 0:
            continue
        n_val = max(1, int(round(n * val_pct))) if n > 1 else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val


def _write_jsonl(path: Path, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_readme(staging_dir: Path, build_meta: dict[str, Any]) -> None:
    sources = ", ".join(build_meta.get("adapters", []))
    total = build_meta.get("packed_total", 0)
    body = (
        "---\n"
        "language:\n"
        "- bn\n"
        "license: other\n"
        "task_categories:\n"
        "- automatic-speech-recognition\n"
        "tags:\n"
        "- bengali\n"
        "- audio\n"
        "- self-supervised\n"
        "---\n\n"
        "# CLAE Bengali Speech Corpus\n\n"
        f"Aggregated Bengali speech corpus packed by `clae_data.pack`.\n\n"
        f"- Sources: {sources}\n"
        f"- Total rows (packed): {total}\n"
        f"- Sample rate: {build_meta.get('target_sr')} Hz mono FLAC\n\n"
        "Manifests under `manifests/`. Audio paths are relative to the repo root.\n"
        "License placeholder: each underlying source retains its original license; verify before redistribution.\n"
    )
    (staging_dir / "README.md").write_text(body, encoding="utf-8")


def _write_build_meta(staging_dir: Path, build_meta: dict[str, Any]) -> None:
    import yaml

    (staging_dir / "build_meta.yaml").write_text(
        yaml.safe_dump(build_meta, sort_keys=False), encoding="utf-8"
    )


def pack_to_dir(
    adapters: list[DatasetAdapter],
    download_root: Path,
    staging_dir: Path,
    target_sr: int = 16000,
    val_pct: float = 0.05,
    asr_probe_pct: float = 0.2,
    seed: int = 42,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    skip_existing: bool = True,
    asr_probe_max_rows: int = 50000,
    asr_probe_val_max_rows: int = 5000,
) -> dict[str, Any]:
    """Download (idempotent), audit, transcode to 16k mono FLAC, split, write manifests.

    Returns the build_meta dict (also written to ``staging_dir/build_meta.yaml``).
    """
    from tqdm import tqdm

    from utils.checkpoint import try_git_hash

    staging_dir = Path(staging_dir)
    download_root = Path(download_root)
    staging_dir.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    # 1) Download + iter_records per adapter.
    raw_counts: dict[str, int] = {}
    all_records: list[Record] = []
    for adapter in adapters:
        print(f"[pack] downloading {adapter.name} -> {download_root}")
        raw_dir = adapter.download(download_root)
        print(f"[pack] iterating records for {adapter.name} from {raw_dir}")
        recs = list(adapter.iter_records(Path(raw_dir)))
        for r in recs:
            r["dataset"] = adapter.name  # belt-and-suspenders
        raw_counts[adapter.name] = len(recs)
        all_records.extend(recs)
        print(f"[pack]   {adapter.name}: {len(recs)} raw records")

    # 2) Audit on raw paths (before transcode), records get duration filled in.
    kept, audit_report = audit_records(
        all_records,
        num_workers=4,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    audited_counts: dict[str, int] = {}
    for r in kept:
        audited_counts[r["dataset"]] = audited_counts.get(r["dataset"], 0) + 1

    # 3) Transcode loop. Serial for simplicity + because torchaudio/soundfile
    # are I/O-bound here and parallel decode adds memory + process-startup
    # overhead that doesn't pay off until well above 100k files. If needed,
    # swap in ProcessPoolExecutor — _transcode_one is pure-ish (no shared state).
    packed: list[Record] = []
    for rec in tqdm(kept, total=len(kept), desc="transcode"):
        new_rec = _transcode_one(rec, staging_dir, target_sr, skip_existing)
        if new_rec is not None:
            packed.append(new_rec)

    packed_counts: dict[str, int] = {}
    for r in packed:
        packed_counts[r["dataset"]] = packed_counts.get(r["dataset"], 0) + 1

    # 4) Deterministic shuffle.
    rng = random.Random(seed)
    rng.shuffle(packed)

    # 5) Per-dataset train/val split.
    train_rows, val_rows = _per_dataset_split(packed, val_pct, rng)
    print(f"[pack] split: {len(train_rows)} train / {len(val_rows)} val")

    # 6) Write the four manifests.
    manifests_dir = staging_dir / "manifests"
    _write_jsonl(manifests_dir / "train.jsonl", train_rows)
    _write_jsonl(manifests_dir / "val.jsonl", val_rows)

    train_text = [r for r in train_rows if r.get("text")]
    val_text = [r for r in val_rows if r.get("text")]

    probe_train_n = min(
        asr_probe_max_rows, max(0, int(round(len(train_text) * asr_probe_pct)))
    )
    probe_val_n = min(asr_probe_val_max_rows, len(val_text))

    # Stable subsample via a dedicated RNG so train.jsonl ordering doesn't leak in.
    probe_rng = random.Random(seed + 1)
    probe_train = list(train_text)
    probe_rng.shuffle(probe_train)
    probe_train = probe_train[:probe_train_n]

    probe_val = list(val_text)
    probe_rng.shuffle(probe_val)
    probe_val = probe_val[:probe_val_n]

    _write_jsonl(manifests_dir / "asr_probe_train.jsonl", probe_train)
    _write_jsonl(manifests_dir / "asr_probe_val.jsonl", probe_val)
    print(
        f"[pack] asr probe: {len(probe_train)} train / {len(probe_val)} val "
        f"(text-labeled pool: {len(train_text)} train / {len(val_text)} val)"
    )

    # 7) build_meta.yaml.
    build_meta: dict[str, Any] = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "git_hash": try_git_hash(),
        "adapters": [a.name for a in adapters],
        "target_sr": int(target_sr),
        "val_pct": float(val_pct),
        "seed": int(seed),
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "raw_counts": raw_counts,
        "audited_counts": audited_counts,
        "packed_counts": packed_counts,
        "audit_report": audit_report,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "asr_probe_train_rows": len(probe_train),
        "asr_probe_val_rows": len(probe_val),
        "packed_total": len(packed),
    }
    _write_build_meta(staging_dir, build_meta)

    # 8) Minimal README dataset card.
    _write_readme(staging_dir, build_meta)

    print(f"[pack] done. staging dir: {staging_dir}")
    return build_meta
