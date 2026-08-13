"""Rigorous speaker-verification benchmark across models.

Larger and more trustworthy than eval_speaker_eer.py: many speakers, EER and
minDCF, evaluated under both mean and mean+std pooling. By default it scores all
pairs; ``--max-trials`` creates a deterministic, stratified predefined subset
that is tractable for paired trial bootstrap.

Pair scoring is vectorized (full cosine matrix), so thousands of utterances and
millions of trials are cheap.

    uv run python -m eval.eval_speaker_verif [--max-utts 2000] [--pools mean,meanstd]

Writes ``runs/eval/speaker_verif.json`` and prints a table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval.repr_bench import (
    DEFAULT_MODELS,
    EVAL_DIR,
    MODEL_ORDER,
    extract_pools,
    load_openslr53_utterances,
)


def _trial_index(
    speakers: np.ndarray,
    *,
    max_trials: int = 0,
    seed: int = 0,
):
    """Stable upper-triangle trials, optionally capped as a fixed trial set."""
    if len(speakers) < 2:
        raise ValueError("Speaker verification needs at least two utterances")
    left, right = np.triu_indices(len(speakers), k=1)
    labels = (speakers[left] == speakers[right]).astype(np.int8)
    if max_trials and len(labels) > max_trials:
        target = np.flatnonzero(labels == 1)
        non_target = np.flatnonzero(labels == 0)
        if not len(target) or not len(non_target):
            raise ValueError("Speaker verification needs target and non-target trials")
        rng = np.random.default_rng(seed)
        target_count = min(len(target), max_trials // 2)
        non_target_count = min(len(non_target), max_trials - target_count)
        remaining = max_trials - target_count - non_target_count
        if remaining:
            extra_target = min(remaining, len(target) - target_count)
            target_count += extra_target
            remaining -= extra_target
            non_target_count += min(remaining, len(non_target) - non_target_count)
        selected = np.sort(np.concatenate([
            rng.choice(target, size=target_count, replace=False),
            rng.choice(non_target, size=non_target_count, replace=False),
        ]))
        left, right, labels = left[selected], right[selected], labels[selected]
    return left.astype(np.int32), right.astype(np.int32), labels


def _trial_scores(X: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine score each predefined trial without rebuilding its identity."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    similarities = Xn @ Xn.T
    return similarities[left, right].astype(np.float32)


def _split_hash(
    item_ids: np.ndarray,
    speakers: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [[str(item_id), str(speaker)] for item_id, speaker in zip(item_ids, speakers)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(left.astype("<i4", copy=False).tobytes())
    digest.update(right.astype("<i4", copy=False).tobytes())
    digest.update(labels.astype(np.int8, copy=False).tobytes())
    return digest.hexdigest()


def eer_and_mindcf(scores: np.ndarray, labels: np.ndarray,
                   p_target: float = 0.01, c_miss: float = 1.0, c_fa: float = 1.0) -> dict:
    from sklearn.metrics import roc_curve

    if not 0.0 < p_target < 1.0:
        raise ValueError("p_target must be strictly between zero and one")
    if len(scores) != len(labels) or len(np.unique(labels)) != 2:
        raise ValueError("EER needs matching scores with positive and negative trials")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[k] + fnr[k]) / 2.0)

    # minDCF over the same operating points.
    dcf = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    norm = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = float(np.min(dcf) / norm)
    return {"eer": eer, "min_dcf": min_dcf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utts", type=int, default=2000)
    ap.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated subset of: " + ",".join(MODEL_ORDER),
    )
    ap.add_argument("--pools", default="mean,meanstd")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help=(
            "Deterministic stratified trial cap (0 keeps all pairs). Use a "
            "bounded fixed set for trial bootstrap."
        ),
    )
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "speaker_verif.json")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    if not models or not pools:
        ap.error("--models and --pools must not be empty")
    if args.max_utts < 2 or args.max_trials < 0 or args.max_trials == 1:
        ap.error("--max-utts must be at least 2; --max-trials is 0 or at least 2")
    utts = load_openslr53_utterances(
        max_utts=args.max_utts, seed=args.data_seed
    )
    item_ids = np.asarray([u.id for u in utts])
    speakers = np.asarray([u.speaker for u in utts])
    left, right, labels = _trial_index(
        speakers,
        max_trials=args.max_trials,
        seed=args.split_seed,
    )
    if len(np.unique(labels)) != 2:
        raise ValueError("Speaker verification trial set needs both classes")
    split_hash = _split_hash(item_ids, speakers, left, right, labels)
    n_spk = len({u.speaker for u in utts})
    bounded_trials = bool(args.max_trials)
    bootstrap_unit = "trial" if bounded_trials else "speaker"
    split_protocol = (
        "deterministic_stratified_predefined_trials"
        if bounded_trials
        else "all_upper_triangle_trials"
    )

    results: dict = {pool: {} for pool in pools}
    artifact_arrays: dict[str, np.ndarray] = {}
    score_sets: list[dict[str, str]] = []
    for name in models:
        pooled = extract_pools(
            name,
            utts,
            ckpt=args.ckpt,
            pools=pools,
            use_cache=not args.no_cache,
        )
        for pool in pools:
            data = pooled[pool]
            scores = _trial_scores(data["X"], left, right)
            m = eer_and_mindcf(scores, labels)
            m.update({"n_pos": int(labels.sum()), "n_neg": int(len(labels) - labels.sum()),
                      "dim": int(data["X"].shape[1])})
            results[pool][name] = m
            score_key = f"scores_{len(score_sets):03d}"
            artifact_arrays[score_key] = scores
            score_sets.append({"key": score_key, "model": name, "pool": pool})

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    trials_out = out.with_name(f"{out.stem}.trials.npz")
    speaker_vocabulary = np.asarray(sorted(set(speakers.tolist())))
    speaker_to_index = {
        str(speaker): index for index, speaker in enumerate(speaker_vocabulary)
    }
    speaker_codes = np.asarray(
        [speaker_to_index[str(speaker)] for speaker in speakers], dtype=np.int32
    )
    artifact_metadata = {
        "schema_version": 1,
        "bootstrap_unit": bootstrap_unit,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "probe_seed": None,
        "split_hash": split_hash,
        "split_protocol": split_protocol,
        "max_trials": args.max_trials,
        "score_sets": score_sets,
        "speaker_fields": "speaker_a and speaker_b index speaker_vocabulary",
        "endpoint_fields": "trial_left and trial_right index utterance_ids",
    }
    np.savez_compressed(
        trials_out,
        utterance_ids=item_ids,
        utterance_speaker=speaker_codes,
        speaker_vocabulary=speaker_vocabulary,
        trial_id=np.arange(len(labels), dtype=np.int64),
        trial_left=left,
        trial_right=right,
        speaker_a=speaker_codes[left],
        speaker_b=speaker_codes[right],
        label=labels,
        metadata=np.asarray(json.dumps(artifact_metadata, sort_keys=True)),
        **artifact_arrays,
    )
    payload = {
        "protocol": split_protocol,
        "n_utts": len(utts),
        "n_speakers": n_spk,
        "p_target": 0.01,
        "pools": pools,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "probe_seed": None,
        "split_hash": split_hash,
        "split_protocol": split_protocol,
        "bootstrap_unit": bootstrap_unit,
        "max_trials": args.max_trials,
        "checkpoint": args.ckpt,
        "predictions_artifact": {
            "path": str(trials_out.resolve()),
            "format": "npz",
            "schema_version": 1,
            "num_trials": int(len(labels)),
            "bootstrap_unit": bootstrap_unit,
            "score_sets": score_sets,
        },
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSpeaker verification  ({len(utts)} utts, {n_spk} speakers)")
    for pool in pools:
        print(f"\n  pool = {pool}")
        print(f"  {'model':<14}{'EER %':>9}{'minDCF':>10}{'dim':>7}")
        print("  " + "-" * 38)
        ranked = sorted(results[pool].items(), key=lambda kv: kv[1]["eer"])
        for name, r in ranked:
            print(f"  {name:<14}{r['eer']*100:>8.2f}{r['min_dcf']:>10.3f}{r['dim']:>7}")
    print(f"\nwrote {out}")
    print(f"wrote {trials_out}")


if __name__ == "__main__":
    main()
