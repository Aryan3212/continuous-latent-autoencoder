"""Fill real durations into JSONL manifests via soundfile.

The housekeeping.py adapters emit ``duration: null``; the probes need real values:
eval_asr's too-long-utterance filter, per-sample CTC input lengths, and the
masked mean/std pooling in iter_embeddings_masked all silently degrade to
worst-case behavior without them. Run this once per manifest on the machine
that has the audio.

Usage:
    uv run python scripts/fill_durations.py M1.jsonl [M2.jsonl ...] [--workers 16]

Each manifest is rewritten in place (atomic replace). Rows whose audio cannot
be read keep their old duration value and are reported. A duration histogram
is printed to guide --segment_seconds / --max_utt_seconds choices.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import soundfile as sf


def _duration(path: str) -> Optional[float]:
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def _pct(sorted_vals: List[float], q: float) -> float:
    i = min(len(sorted_vals) - 1, max(0, int(round(q / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def fill(manifest: str, workers: int) -> None:
    mpath = pathlib.Path(manifest).resolve()
    rows = []
    with open(mpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Relative audio_filepath resolves against the manifest's parent directory,
    # matching AudioDataset.
    paths = []
    for r in rows:
        p = r.get("audio_filepath", "")
        paths.append(p if os.path.isabs(p) else str(mpath.parent / p))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        durs = list(ex.map(_duration, paths))

    failed = 0
    for r, d in zip(rows, durs):
        if d is None:
            failed += 1
        else:
            r["duration"] = d

    tmp = mpath.with_suffix(mpath.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, mpath)

    ok = sorted(d for d in durs if d is not None)
    print(f"[fill] {mpath.name}: {len(rows)} rows, {len(ok)} filled, {failed} unreadable (left untouched)")
    if ok:
        pcts = "  ".join(f"p{q}={_pct(ok, q):.2f}" for q in (5, 25, 50, 75, 95, 99))
        print(f"[fill]   duration (s): {pcts}  max={ok[-1]:.2f}")
        for cut in (2.5, 5.0, 10.0, 15.0):
            n = sum(1 for d in ok if d <= cut)
            print(f"[fill]   <= {cut:g}s: {n} ({100.0 * n / len(ok):.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    for m in args.manifests:
        fill(m, args.workers)


if __name__ == "__main__":
    main()
