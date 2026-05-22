"""Parallel sanity check over raw audio paths before packing."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from clae_data.schema import Record


def _audit_one(args: tuple[int, dict[str, Any], float, float]) -> dict[str, Any]:
    """Worker: probe one audio file with soundfile.info.

    Returns a dict with at least ``index`` and ``status``. ``status`` is one of
    ``ok``, ``missing``, ``too_short``, ``too_long``, ``empty``, ``corrupt``.
    On ``ok`` the dict also carries ``duration`` (seconds, from sf.info).
    """
    import soundfile as sf

    idx, rec, min_duration, max_duration = args
    path = rec.get("audio_filepath")
    if not path or not Path(path).exists():
        return {"index": idx, "status": "missing", "path": path}
    try:
        info = sf.info(path)
    except Exception as e:
        return {"index": idx, "status": "corrupt", "path": path, "error": str(e)}
    if info.frames == 0:
        return {"index": idx, "status": "empty", "path": path}
    dur = float(info.duration)
    if dur < min_duration:
        return {"index": idx, "status": "too_short", "path": path, "duration": dur}
    if dur > max_duration:
        return {"index": idx, "status": "too_long", "path": path, "duration": dur}
    return {"index": idx, "status": "ok", "path": path, "duration": dur}


def audit_records(
    records: Iterable[Record],
    num_workers: int = 4,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> tuple[list[Record], dict[str, Any]]:
    """Probe every record's audio file in parallel and drop bad rows.

    Returns ``(kept_records, report)``. The kept records have their ``duration``
    field overwritten with the measured value. The report dict has per-status
    counts plus the parameters used.
    """
    from tqdm import tqdm

    rec_list: list[Record] = list(records)
    work = [
        (i, dict(r), min_duration, max_duration) for i, r in enumerate(rec_list)
    ]

    results: list[dict[str, Any]] = []
    if num_workers <= 1:
        for w in tqdm(work, total=len(work), desc="audit"):
            results.append(_audit_one(w))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for res in tqdm(
                ex.map(_audit_one, work, chunksize=64),
                total=len(work),
                desc="audit",
            ):
                results.append(res)

    counts: dict[str, int] = {}
    kept: list[Record] = []
    for res in results:
        st = res["status"]
        counts[st] = counts.get(st, 0) + 1
        if st == "ok":
            rec = rec_list[res["index"]]
            rec["duration"] = res["duration"]
            kept.append(rec)

    report = {
        "total": len(rec_list),
        "kept": len(kept),
        "counts": counts,
        "min_duration": min_duration,
        "max_duration": max_duration,
    }
    print("[audit] summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"[audit] kept {len(kept)} / {len(rec_list)} rows")
    return kept, report
