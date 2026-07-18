"""Materialize the processed Hugging Face SUBESCO dataset for local evals.

The Hub dataset is stored as Parquet audio rows rather than a downloadable ZIP.
This script downloads the published split, writes lossless WAV files under
``datasets/SUBESCO/audio/``, and preserves every row's labels in
``datasets/SUBESCO/metadata.tsv``. The emotion evaluators discover WAV files
recursively, so no manifest generation is required.

Usage:
    uv run python scripts/download_subesco.py
    uv run python scripts/download_subesco.py --out-dir /data/SUBESCO
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ID = "sajid73/SUBESCO-audio-dataset"
DEFAULT_OUT = Path("datasets/SUBESCO")
MARKER = ".subesco_download.json"


def _safe_name(value: str, fallback: str) -> str:
    name = Path(value).name.strip() or fallback
    return name if name.lower().endswith(".wav") else f"{name}.wav"


def _label_name(dataset: Any, label: Any) -> str:
    feature = dataset.features.get("label")
    if hasattr(feature, "int2str") and isinstance(label, int):
        return str(feature.int2str(label))
    return str(label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Replace an existing completed materialization.")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    marker = out_dir / MARKER
    if marker.exists() and not args.force:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        print(f"SUBESCO already materialized at {out_dir} ({payload['rows']} rows); use --force to replace it.")
        return
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)

    from datasets import load_dataset
    import soundfile as sf

    dataset = load_dataset(args.repo_id, split=args.split, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.tsv"
    seen: set[str] = set()
    fields = [
        "file_name", "audio_filepath", "transcription", "speaker_id", "speaker_name",
        "speaker_gender", "sentence_no", "repetation_no", "label",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(dataset):
            audio = row["audio"]
            name = _safe_name(str(row.get("file name") or row.get("file_name") or ""), f"subesco_{index:05d}.wav")
            if name in seen:
                raise ValueError(f"Duplicate audio filename in dataset: {name}")
            seen.add(name)
            dest = audio_dir / name
            sf.write(dest, audio["array"], int(audio["sampling_rate"]), subtype="PCM_16")
            writer.writerow({
                "file_name": name,
                "audio_filepath": str(dest.relative_to(out_dir)),
                "transcription": row.get("transcription", ""),
                "speaker_id": row.get("speaker_id", ""),
                "speaker_name": row.get("speaker_name", ""),
                "speaker_gender": row.get("speaker_gender", ""),
                "sentence_no": row.get("sentence_no", ""),
                "repetation_no": row.get("repetation_no", ""),
                "label": _label_name(dataset, row.get("label", "")),
            })
            if (index + 1) % 500 == 0:
                print(f"materialized {index + 1}/{len(dataset)} clips", flush=True)

    marker.write_text(json.dumps({
        "repo_id": args.repo_id, "split": args.split, "rows": len(dataset),
        "audio_dir": "audio", "metadata": "metadata.tsv",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(dataset)} clips to {audio_dir}")
    print(f"wrote labels to {metadata_path}")


if __name__ == "__main__":
    main()
