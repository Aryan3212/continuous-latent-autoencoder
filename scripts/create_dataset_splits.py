#!/usr/bin/env python3
"""
Create reproducible train/val/test splits from a directory of audio files.
This script scans for audio files, calculates their durations, filters out
short files, and creates JSONL manifests.

Usage:
    uv run python scripts/create_dataset_splits.py \
        --data_dir /path/to/wavs \
        --out_dir data/manifests \
        --min_duration 1.0 \
        --val_frac 0.05 \
        --test_frac 0.05 \
        --seed 42
"""

import argparse
import glob
import json
import os
import random
import pathlib
import soundfile as sf
import tqdm
from typing import List, Dict, Any

def get_audio_files(root: str, extensions: List[str] = [".wav", ".mp3", ".flac", ".ogg"]) -> List[str]:
    files = []
    root_path = pathlib.Path(root)
    for ext in extensions:
        # Recursive glob
        files.extend([str(p) for p in root_path.rglob(f"*{ext}")])
        files.extend([str(p) for p in root_path.rglob(f"*{ext.upper()}")])
    return sorted(list(set(files)))

def get_duration(path: str) -> float:
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        # Fallback for some formats or errors
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Root directory containing audio files")
    ap.add_argument("--out_dir", required=True, help="Output directory for manifests")
    ap.add_argument("--min_duration", type=float, default=2.0, help="Skip files shorter than this (seconds)")
    ap.add_argument("--val_frac", type=float, default=0.05, help="Fraction of data for validation")
    ap.add_argument("--test_frac", type=float, default=0.05, help="Fraction of data for testing")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--limit", type=int, default=None, help="Limit total files (for testing)")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.data_dir}...")
    files = get_audio_files(args.data_dir)
    print(f"Found {len(files)} files.")

    if args.limit:
        files = files[:args.limit]
        print(f"Limiting to {args.limit} files.")

    # Shuffle deterministically
    random.shuffle(files)

    valid_items: List[Dict[str, Any]] = []
    
    print("Processing files (calculating durations)...")
    for fpath in tqdm.tqdm(files):
        dur = get_duration(fpath)
        if dur >= args.min_duration:
            valid_items.append({
                "audio_filepath": os.path.abspath(fpath),
                "duration": dur
            })
    
    print(f"Kept {len(valid_items)} files after filtering (min_duration={args.min_duration}s).")

    # Split
    n_total = len(valid_items)
    n_val = int(n_total * args.val_frac)
    n_test = int(n_total * args.test_frac)
    n_train = n_total - n_val - n_test

    train_items = valid_items[:n_train]
    val_items = valid_items[n_train : n_train + n_val]
    test_items = valid_items[n_train + n_val :]

    print(f"Splits: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")

    def write_manifest(items, name):
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"Wrote {path}")

    write_manifest(train_items, "train")
    write_manifest(val_items, "val")
    write_manifest(test_items, "test")

    # Write split info
    split_info = {
        "seed": args.seed,
        "source_dir": os.path.abspath(args.data_dir),
        "counts": {
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items)
        },
        "total_files_scanned": len(files),
        "total_files_valid": len(valid_items)
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

if __name__ == "__main__":
    main()
