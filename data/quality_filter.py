from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np

from data.dataset import _load_audio


def _snr_proxy_db(x: np.ndarray, sr: int) -> float:
    # Frame-based proxy: compare full RMS to low-energy RMS.
    frame = int(0.03 * sr)
    hop = int(0.01 * sr)
    rms = []
    for i in range(max(1, (len(x) - frame) // hop + 1)):
        w = x[i * hop : i * hop + frame]
        rms.append(float(np.sqrt(np.mean(w * w) + 1e-12)))
    rms = np.asarray(rms, dtype=np.float32)
    if rms.size == 0:
        return 0.0
    hi = float(np.mean(rms))
    lo = float(np.mean(np.sort(rms)[: max(1, int(0.1 * rms.size))]))
    return float(20.0 * np.log10((hi + 1e-9) / (lo + 1e-9)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--reject_manifest", default=None)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--max_clip_frac", type=float, default=0.005)
    ap.add_argument("--max_silence_frac", type=float, default=0.40)
    ap.add_argument("--silence_thr", type=float, default=1e-4)
    ap.add_argument("--min_snr_proxy_db", type=float, default=10.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in pathlib.Path(args.in_manifest).read_text().splitlines() if l.strip()]
    out_path = pathlib.Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rej_path = pathlib.Path(args.reject_manifest) if args.reject_manifest else None
    if rej_path:
        rej_path.parent.mkdir(parents=True, exist_ok=True)

    sr = int(args.sample_rate)
    keep, reject = 0, 0
    stats = {"clip_frac": [], "silence_frac": [], "snr_proxy_db": []}

    out_f = out_path.open("w", encoding="utf-8")
    rej_f = rej_path.open("w", encoding="utf-8") if rej_path else None
    try:
        for r in rows:
            start = r.get("start")
            dur = r.get("duration")
            wav = _load_audio(
                r["audio_filepath"],
                sr,
                start_sec=float(start) if start is not None else None,
                duration_sec=float(dur) if dur is not None else None,
            ).cpu().numpy()
            clip_frac = float(np.mean(np.abs(wav) >= 0.999))
            silence_frac = float(np.mean(np.abs(wav) <= float(args.silence_thr)))
            snr_db = _snr_proxy_db(wav, sr)

            ok = (
                clip_frac <= float(args.max_clip_frac)
                and silence_frac <= float(args.max_silence_frac)
                and snr_db >= float(args.min_snr_proxy_db)
            )
            row = dict(r)
            row["clip_frac"] = clip_frac
            row["silence_frac"] = silence_frac
            row["snr_proxy_db"] = snr_db
            if ok:
                keep += 1
                stats["clip_frac"].append(clip_frac)
                stats["silence_frac"].append(silence_frac)
                stats["snr_proxy_db"].append(snr_db)
                out_f.write(json.dumps(row) + "\n")
            else:
                reject += 1
                if rej_f:
                    rej_f.write(json.dumps(row) + "\n")
    finally:
        out_f.close()
        if rej_f:
            rej_f.close()

    def _summ(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"min": 0.0, "med": 0.0, "max": 0.0}
        xs2 = np.asarray(xs, dtype=np.float32)
        return {"min": float(xs2.min()), "med": float(np.median(xs2)), "max": float(xs2.max())}

    report = {
        "kept": keep,
        "rejected": reject,
        "clip_frac": _summ(stats["clip_frac"]),
        "silence_frac": _summ(stats["silence_frac"]),
        "snr_proxy_db": _summ(stats["snr_proxy_db"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

