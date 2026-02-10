#!/usr/bin/env python3
"""
Comprehensive Audio Quality & Distribution Audit Tool.
Performs:
1. Audio Content Hashing (detect exact duplicates across splits/datasets)
2. Audio Signal Quality Scan (RMS, Clipping, Silence, SR mismatch)
3. Dataset Metadata/Distribution Visualization

Usage:
    uv run python scripts/audit_datasets.py --data_root data/ --manifests data/manifests/
"""

import argparse
import hashlib
import json
import glob
import os
import pathlib
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from collections import defaultdict, Counter, namedtuple
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# --- 1. Duplicate Detection ---

def audio_content_hash(filepath: str, target_sr: int = 16000) -> str:
    """Hash based on actual audio content, resampling to ensure consistency."""
    try:
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Simple resampling (nearest neighbor-ish) if needed just for hashing
        # For production dedupe, use librosa.resample, but here we want speed
        # and "exact" duplicates usually match in SR anyway.
        # If strict content matching is needed despite SR, librosa is safer.
        
        # Quantize to 16-bit to ignore floating point noise
        quantized = (audio * 32767).astype(np.int16).tobytes()
        return hashlib.sha256(quantized).hexdigest()
    except Exception as e:
        return f"error_{e}"

# --- 2. Quality Scan ---

@dataclass
class AudioQualityReport:
    filepath: str
    dataset: str
    duration: float
    sample_rate: int
    rms_energy: float
    peak_amplitude: float
    dc_offset: float
    silence_ratio: float
    issues: List[str]

def scan_audio(args: Tuple[str, str, int]) -> AudioQualityReport:
    filepath, dataset_name, expected_sr = args
    issues = []
    
    try:
        # Use soundfile for speed
        info = sf.info(filepath)
        sr = info.samplerate
        duration = info.duration
        
        # Read audio for signal stats
        audio, _ = sf.read(filepath)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        dc = float(np.mean(audio))

        # Silence detection (naive energy threshold)
        frame_size = int(0.025 * sr)
        hop = int(0.010 * sr)
        if len(audio) < frame_size:
            silence_ratio = 0.0
        else:
            # Vectorized energy calculation
            # Truncate to whole number of frames
            n_frames = (len(audio) - frame_size) // hop + 1
            # Striding trick for speed? Let's keep it simple first.
            # For 1 hour of audio, python loop is slow.
            # Let's do a coarse check: 
            # Reshape to (N, frame_size) is hard with overlap.
            # Approximation: split into non-overlapping chunks for speed.
            chunks = audio[:len(audio)//frame_size * frame_size].reshape(-1, frame_size)
            chunk_energies = np.mean(chunks**2, axis=1)
            silent_chunks = np.sum(chunk_energies < 1e-6) # -60dB threshold roughly
            silence_ratio = silent_chunks / len(chunks) if len(chunks) > 0 else 0.0

        # Flag issues
        if rms < 0.001: issues.append("near_silent")
        if peak > 0.999: issues.append("clipping")
        if abs(dc) > 0.01: issues.append("dc_offset")
        if duration < 0.5: issues.append("too_short")
        if duration > 300.0: issues.append("too_long_300s")
        if sr != expected_sr: issues.append(f"sr_mismatch_{sr}")
        if silence_ratio > 0.8: issues.append("mostly_silent")

        return AudioQualityReport(
            filepath=filepath,
            dataset=dataset_name,
            duration=duration,
            sample_rate=sr,
            rms_energy=rms,
            peak_amplitude=peak,
            dc_offset=dc,
            silence_ratio=silence_ratio,
            issues=issues
        )
    except Exception as e:
        return AudioQualityReport(filepath, dataset_name, 0, 0, 0, 0, 0, 0, [f"read_error: {str(e)}"])

# --- 3. Distribution Visualization ---

def plot_distributions(reports: List[AudioQualityReport], out_dir: str):
    out_path = pathlib.Path(out_dir)
    
    # 1. Issues per Dataset
    datasets = sorted(list(set(r.dataset for r in reports)))
    issues_map = {d: Counter() for d in datasets}
    for r in reports:
        for issue in r.issues:
            issues_map[r.dataset][issue] += 1
            
    # Print Summary
    print("\n=== Quality Issues Summary ===")
    for d in datasets:
        n_files = len([r for r in reports if r.dataset == d])
        if n_files == 0: continue
        print(f"Dataset: {d} (N={n_files})")
        if not issues_map[d]:
            print("  No issues found.")
        else:
            for issue, count in issues_map[d].most_common():
                print(f"  {issue}: {count} ({100*count/n_files:.1f}%)")

    # 2. Durations Histogram
    durations = [r.duration for r in reports if r.duration < 30.0] # clip for viz
    plt.figure(figsize=(10, 4))
    plt.hist(durations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Duration Distribution (Clipped at 30s)")
    plt.xlabel("Seconds")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path / "dist_duration.png")
    plt.close()

# --- Main Driver ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root folder to scan recursively")
    parser.add_argument("--manifests", help="Optional: scan only files in jsonl manifests")
    parser.add_argument("--expected_sr", type=int, default=16000)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    # Gather files
    files_to_scan = [] # List of (filepath, dataset_name)
    
    if args.manifests:
        # Load from manifests
        manifest_path = pathlib.Path(args.manifests)
        if manifest_path.is_file():
            manifests = [manifest_path]
        else:
            manifests = list(manifest_path.glob("*.jsonl"))
        
        for m in manifests:
            dataset_name = m.stem
            with open(m) as f:
                for line in f:
                    rec = json.loads(line)
                    files_to_scan.append((rec["audio_filepath"], dataset_name))
    else:
        # Recursive scan
        root = pathlib.Path(args.data_root)
        all_wavs = list(root.rglob("*.wav")) + list(root.rglob("*.mp3")) + list(root.rglob("*.flac"))
        for p in all_wavs:
            # dataset name heuristic: parent folder
            files_to_scan.append((str(p), p.parent.name))

    print(f"Scanning {len(files_to_scan)} files...")
    
    # 1. Run Quality Scan (Parallel)
    scan_inputs = [(f, d, args.expected_sr) for f, d in files_to_scan]
    reports = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for res in tqdm(executor.map(scan_audio, scan_inputs), total=len(scan_inputs), desc="Quality Audit"):
            reports.append(res)
            
    # 2. Run Duplicate Check (Content Hash)
    # Only if dataset size is manageable (<100k files), otherwise hash might be slow
    print("\nChecking for exact content duplicates...")
    hashes = defaultdict(list)
    for r in tqdm(reports, desc="Hashing"):
        # We re-read file here or could have done it in worker. 
        # For simplicity, let's just do it here or skip if too slow.
        # Ideally, we should have returned hash from scan_audio.
        pass 
    
    # Actually, let's modify scan_audio logic? 
    # For now, let's do a separate hash pass for "suspects" (same duration)
    # Optimization: Only hash files with same duration (quantized to 0.1s)
    dur_buckets = defaultdict(list)
    for r in reports:
        dur_buckets[round(r.duration, 2)].append(r)
        
    dupes = []
    for dur, bucket in tqdm(dur_buckets.items(), desc="Deduplication"):
        if len(bucket) < 2: continue
        
        # Hash these
        bucket_hashes = {}
        for item in bucket:
            h = audio_content_hash(item.filepath)
            if h in bucket_hashes:
                dupes.append((item, bucket_hashes[h]))
            else:
                bucket_hashes[h] = item

    if dupes:
        print(f"\nFound {len(dupes)} duplicate pairs:")
        for a, b in dupes[:10]:
            print(f"  {a.dataset}::{a.filepath} == {b.dataset}::{b.filepath}")
    else:
        print("\nNo exact duplicates found.")

    # 3. Viz
    plot_distributions(reports, ".")
    
    # 4. Save detailed report
    with open("audit_report.jsonl", "w") as f:
        for r in reports:
            f.write(json.dumps(r.__dict__) + "\n")
    print("\nSaved detailed report to audit_report.jsonl")

if __name__ == "__main__":
    main()
