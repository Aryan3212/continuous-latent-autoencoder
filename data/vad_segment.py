from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np

from data.dataset import _load_audio  # reuse IO + resampling


def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n = max(1, (len(x) - frame) // hop + 1)
    out = np.empty((n,), dtype=np.float32)
    for i in range(n):
        s = i * hop
        w = x[s : s + frame]
        out[i] = float(np.sqrt(np.mean(w * w) + 1e-12))
    return out


def _segments_from_mask(mask: np.ndarray, hop_s: float, min_dur: float, pad: float) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        start = max(0.0, i * hop_s - pad)
        end = j * hop_s + pad
        if end - start >= min_dur:
            segs.append((start, end))
        i = j
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True, help="JSONL with at least {audio_filepath}")
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--frame_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--thr_db", type=float, default=-35.0, help="RMS dBFS threshold relative to full-scale=1.0")
    ap.add_argument("--min_dur", type=float, default=2.0)
    ap.add_argument("--max_dur", type=float, default=8.0)
    ap.add_argument("--pad", type=float, default=0.15)
    args = ap.parse_args()

    rows = [json.loads(l) for l in pathlib.Path(args.in_manifest).read_text().splitlines() if l.strip()]
    out_path = pathlib.Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sr = int(args.sample_rate)
    frame = int(round(sr * (args.frame_ms / 1000.0)))
    hop = int(round(sr * (args.hop_ms / 1000.0)))
    hop_s = hop / sr
    thr = 10 ** (args.thr_db / 20.0)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            wav = _load_audio(r["audio_filepath"], sr).cpu().numpy()
            rms = _frame_rms(wav, frame=frame, hop=hop)
            speech = rms > thr
            segs = _segments_from_mask(speech, hop_s=hop_s, min_dur=float(args.min_dur), pad=float(args.pad))
            for start, end in segs:
                dur = end - start
                if dur > float(args.max_dur):
                    # chunk long segments
                    n_chunks = int(math.ceil(dur / float(args.max_dur)))
                    for k in range(n_chunks):
                        s = start + k * float(args.max_dur)
                        e = min(end, s + float(args.max_dur))
                        if e - s >= float(args.min_dur):
                            row = dict(r)
                            row["start"] = float(s)
                            row["duration"] = float(e - s)
                            f.write(json.dumps(row) + "\n")
                            written += 1
                else:
                    row = dict(r)
                    row["start"] = float(start)
                    row["duration"] = float(dur)
                    f.write(json.dumps(row) + "\n")
                    written += 1

    print(json.dumps({"segments_written": written, "thr_db": args.thr_db, "min_dur": args.min_dur, "max_dur": args.max_dur}, indent=2))


if __name__ == "__main__":
    main()

