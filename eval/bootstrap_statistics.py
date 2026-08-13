"""Paired bootstrap statistics for standardized TACL evaluation artifacts.

The command accepts one or more evaluator summaries as
``--entry TASK SYSTEM SEED SUMMARY``. ``SEED`` is an integer for a trained
probe and ``none`` for deterministic evaluations.  Evaluator summaries may
contain records inline or point to a JSON/JSONL artifact:

* reconstruction: ``per_item`` or ``per_item_artifact``; every row has
  ``item_id`` and a numeric ``metrics`` mapping.
* ASR: ``predictions``/``predictions_artifact`` (also accepted under ``dev``);
  every row has ``item_id``, ``reference``, and ``hypothesis``.
* classification: ``predictions`` or ``predictions_artifact``; every row has
  ``item_id``, ``speaker_id``, ``gold``, and ``prediction``.
* verification: ``trials`` or ``trials_artifact``; every row has ``trial_id``,
  ``label``, and ``score``.  ``bootstrap_unit`` is either ``speaker`` (rows
  also have enrollment/test speaker IDs) or ``trial``.

Unit identities and labels are checked across every system and probe seed
before any resampling.  Probe-seed variation and sample uncertainty are
reported separately: the former is the sample standard deviation over saved
probe runs; the latter is a paired percentile interval using joint resamples.
Bootstrap never retrains a probe.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


TASKS = {"reconstruction", "asr", "classification", "verification"}


def _task_kind(task: str) -> str:
    """Allow named task instances such as ``classification:emotion``."""
    kind = task.split(":", 1)[0]
    if kind not in TASKS:
        raise ValueError(f"unknown task: {task}")
    return kind


@dataclass(frozen=True)
class EvaluationEntry:
    task: str
    system: str
    seed: str
    records: list[dict[str, Any]]
    bootstrap_unit: str | None = None
    classification_metric: str | None = None
    verification_arrays: "VerificationArrays | None" = None


@dataclass(frozen=True)
class VerificationArrays:
    trial_ids: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    bootstrap_unit: str
    speaker_a: np.ndarray | None = None
    speaker_b: np.ndarray | None = None
    speaker_vocabulary: np.ndarray | None = None
    trial_left: np.ndarray | None = None
    trial_right: np.ndarray | None = None
    utterance_ids: np.ndarray | None = None
    utterance_speaker: np.ndarray | None = None
    split_hash: str | None = None


def sample_standard_deviation(values: Sequence[float]) -> float | None:
    """Return the sample SD, or ``None`` when there is only one run."""
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def percentile_interval(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot form an interval from no bootstrap values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [tail, 1.0 - tail])
    return float(low), float(high)


def _seed_sort_key(seed: str) -> tuple[int, int | str]:
    try:
        return (0, int(seed))
    except ValueError:
        return (1, seed)


def _index_records(
    records: Sequence[Mapping[str, Any]], id_key: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in records:
        value = row.get(id_key)
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise ValueError(f"every record needs a string/integer {id_key}")
        unit_id = str(value)
        if unit_id in indexed:
            raise ValueError(f"duplicate {id_key}: {unit_id}")
        normalized = dict(row)
        if "gold" not in normalized and "gold_label" in normalized:
            normalized["gold"] = normalized["gold_label"]
        if "prediction" not in normalized and "predicted_label" in normalized:
            normalized["prediction"] = normalized["predicted_label"]
        indexed[unit_id] = normalized
    if not indexed:
        raise ValueError("evaluation artifact contains no records")
    return indexed


def _organized(
    entries: Sequence[EvaluationEntry],
) -> tuple[list[str], list[str], dict[str, dict[str, list[dict[str, Any]]]]]:
    systems: list[str] = []
    seeds: list[str] = []
    data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entry in entries:
        if entry.system not in data:
            systems.append(entry.system)
            data[entry.system] = {}
        if entry.seed in data[entry.system]:
            raise ValueError(
                f"duplicate task/system/seed entry: {entry.task}/{entry.system}/{entry.seed}"
            )
        data[entry.system][entry.seed] = entry.records
        if entry.seed not in seeds:
            seeds.append(entry.seed)
    if len(systems) < 2:
        raise ValueError("paired bootstrap requires at least two systems")
    expected_seeds = set(data[systems[0]])
    for system in systems[1:]:
        if set(data[system]) != expected_seeds:
            raise ValueError("all systems must provide the same probe-seed set")
    return systems, sorted(seeds, key=_seed_sort_key), data


def _aligned_records(
    entries: Sequence[EvaluationEntry],
    id_key: str,
    invariant_keys: Sequence[str],
) -> tuple[
    list[str],
    list[str],
    list[str],
    dict[str, dict[str, dict[str, dict[str, Any]]]],
]:
    systems, seeds, raw = _organized(entries)
    indexed: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    canonical_ids: set[str] | None = None
    invariants: dict[str, tuple[Any, ...]] = {}
    for system in systems:
        indexed[system] = {}
        for seed in seeds:
            rows = _index_records(raw[system][seed], id_key)
            row_ids = set(rows)
            if canonical_ids is None:
                canonical_ids = row_ids
                invariants = {
                    unit_id: tuple(rows[unit_id].get(key) for key in invariant_keys)
                    for unit_id in row_ids
                }
            elif row_ids != canonical_ids:
                missing = sorted(canonical_ids - row_ids)[:5]
                extra = sorted(row_ids - canonical_ids)[:5]
                raise ValueError(
                    "unit IDs differ across systems/seeds "
                    f"(missing={missing}, extra={extra})"
                )
            for unit_id, expected in invariants.items():
                actual = tuple(rows[unit_id].get(key) for key in invariant_keys)
                if actual != expected:
                    raise ValueError(
                        f"invariant fields {list(invariant_keys)} differ for {unit_id}"
                    )
            indexed[system][seed] = rows
    assert canonical_ids is not None
    return systems, seeds, sorted(canonical_ids), indexed


def _summary(
    systems: Sequence[str],
    seeds: Sequence[str],
    point_by_system_seed: Mapping[str, Mapping[str, float]],
    bootstrap_by_system: Mapping[str, Sequence[float]],
    confidence: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {"systems": {}, "paired_differences": {}}
    means: dict[str, float] = {}
    for system in systems:
        seed_values = [float(point_by_system_seed[system][seed]) for seed in seeds]
        seed_mean = float(np.mean(seed_values))
        means[system] = seed_mean
        low, high = percentile_interval(bootstrap_by_system[system], confidence)
        output["systems"][system] = {
            "seed_values": {
                seed: float(point_by_system_seed[system][seed]) for seed in seeds
            },
            "seed_mean": seed_mean,
            "seed_sample_sd": sample_standard_deviation(seed_values),
            "bootstrap_ci": {"low": low, "high": high},
        }
    for first, second in itertools.combinations(systems, 2):
        differences = (
            np.asarray(bootstrap_by_system[second], dtype=np.float64)
            - np.asarray(bootstrap_by_system[first], dtype=np.float64)
        )
        low, high = percentile_interval(differences.tolist(), confidence)
        key = f"{second} - {first}"
        output["paired_differences"][key] = {
            "system_a": first,
            "system_b": second,
            "estimate_b_minus_a": means[second] - means[first],
            "bootstrap_ci": {"low": low, "high": high},
        }
    return output


def bootstrap_reconstruction(
    entries: Sequence[EvaluationEntry],
    *,
    replicates: int,
    bootstrap_seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    systems, seeds, unit_ids, data = _aligned_records(entries, "item_id", ())
    first = data[systems[0]][seeds[0]][unit_ids[0]].get("metrics")
    if not isinstance(first, Mapping):
        raise ValueError("reconstruction rows need a metrics mapping")
    metric_names = sorted(
        key for key, value in first.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if not metric_names:
        raise ValueError("reconstruction artifact has no numeric per-item metrics")
    for system in systems:
        for seed in seeds:
            for unit_id in unit_ids:
                metrics = data[system][seed][unit_id].get("metrics")
                if not isinstance(metrics, Mapping):
                    raise ValueError(f"missing metrics mapping for {unit_id}")
                for name in metric_names:
                    value = metrics.get(name)
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(
                            f"metric {name} is incomplete/non-finite for {system}/{seed}/{unit_id}"
                        )

    rng = np.random.default_rng(bootstrap_seed)
    result: dict[str, Any] = {"bootstrap_unit": "utterance", "metrics": {}}
    n = len(unit_ids)
    sampled_indices = [
        rng.integers(0, n, size=n) for _ in range(replicates)
    ]
    for name in metric_names:
        values = np.asarray(
            [
                [
                    [
                        float(data[system][seed][unit_id]["metrics"][name])
                        for unit_id in unit_ids
                    ]
                    for seed in seeds
                ]
                for system in systems
            ],
            dtype=np.float64,
        )
        point = {
            system: {
                seed: float(values[system_index, seed_index].mean())
                for seed_index, seed in enumerate(seeds)
            }
            for system_index, system in enumerate(systems)
        }
        boot_matrix = np.asarray(
            [values[:, :, sampled].mean(axis=(1, 2)) for sampled in sampled_indices],
            dtype=np.float64,
        )
        boot = {
            system: boot_matrix[:, system_index].tolist()
            for system_index, system in enumerate(systems)
        }
        result["metrics"][name] = _summary(
            systems, seeds, point, boot, confidence
        )
    return result


def _asr_error_counts(reference: str, hypothesis: str, name: str) -> tuple[int, int]:
    """Return additive edit errors and reference units for one utterance."""
    from jiwer import process_characters, process_words

    output = (
        process_words(reference, hypothesis)
        if name == "wer"
        else process_characters(reference, hypothesis)
    )
    errors = int(output.substitutions + output.deletions + output.insertions)
    reference_units = int(output.hits + output.substitutions + output.deletions)
    return errors, reference_units


def _asr_rate(counts: Sequence[tuple[int, int]]) -> float:
    errors = sum(item[0] for item in counts)
    reference_units = sum(item[1] for item in counts)
    if reference_units == 0:
        # This matches jiwer's empty-reference corpus convention.
        return float(errors > 0)
    return float(errors / reference_units)


def bootstrap_asr(
    entries: Sequence[EvaluationEntry],
    *,
    replicates: int,
    bootstrap_seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    systems, seeds, unit_ids, data = _aligned_records(
        entries, "item_id", ("reference",)
    )
    for system in systems:
        for seed in seeds:
            for unit_id in unit_ids:
                row = data[system][seed][unit_id]
                if not isinstance(row.get("reference"), str) or not isinstance(
                    row.get("hypothesis"), str
                ):
                    raise ValueError("ASR rows need string reference and hypothesis fields")

    rng = np.random.default_rng(bootstrap_seed)
    n = len(unit_ids)
    sampled_indices = [
        rng.integers(0, n, size=n) for _ in range(replicates)
    ]
    result: dict[str, Any] = {"bootstrap_unit": "utterance", "metrics": {}}
    for name in ("wer", "cer"):
        counts = {
            system: {
                seed: [
                    _asr_error_counts(
                        str(data[system][seed][unit_id]["reference"]),
                        str(data[system][seed][unit_id]["hypothesis"]),
                        name,
                    )
                    for unit_id in unit_ids
                ]
                for seed in seeds
            }
            for system in systems
        }
        point = {
            system: {
                seed: _asr_rate(counts[system][seed])
                for seed in seeds
            }
            for system in systems
        }
        error_array = np.asarray(
            [
                [
                    [item[0] for item in counts[system][seed]]
                    for seed in seeds
                ]
                for system in systems
            ],
            dtype=np.int64,
        )
        reference_array = np.asarray(
            [
                [
                    [item[1] for item in counts[system][seed]]
                    for seed in seeds
                ]
                for system in systems
            ],
            dtype=np.int64,
        )
        boot_matrix = np.empty((replicates, len(systems)), dtype=np.float64)
        for replicate, sampled in enumerate(sampled_indices):
            errors = error_array[:, :, sampled].sum(axis=2)
            references = reference_array[:, :, sampled].sum(axis=2)
            rates = np.divide(
                errors,
                references,
                out=(errors > 0).astype(np.float64),
                where=references > 0,
            )
            boot_matrix[replicate] = rates.mean(axis=1)
        boot = {
            system: boot_matrix[:, index].tolist()
            for index, system in enumerate(systems)
        }
        result["metrics"][name] = _summary(
            systems, seeds, point, boot, confidence
        )
    return result


def _fold_classification_metrics(
    gold: np.ndarray,
    predicted: np.ndarray,
    folds: np.ndarray,
    weights: np.ndarray,
    score_name: str,
) -> tuple[float, float]:
    """Match evaluator scores: compute each fold, then average folds equally."""
    if score_name not in {"accuracy", "balanced_accuracy"}:
        raise ValueError(f"unsupported classification score: {score_name}")
    if not (len(gold) == len(predicted) == len(folds) == len(weights)):
        raise ValueError("classification arrays must align")
    fold_scores: list[float] = []
    fold_f1s: list[float] = []
    for fold in np.unique(folds):
        in_fold = folds == fold
        active = in_fold & (weights > 0)
        if not np.any(active):
            raise ValueError(f"speaker resample emptied required fold {fold}")
        fold_gold = gold[active]
        fold_predicted = predicted[active]
        fold_weights = weights[active]
        if score_name == "accuracy":
            score = float(
                fold_weights[fold_gold == fold_predicted].sum()
                / fold_weights.sum()
            )
        else:
            recalls: list[float] = []
            for label in np.unique(fold_gold):
                is_label = fold_gold == label
                recalls.append(
                    float(
                        fold_weights[is_label & (fold_predicted == label)].sum()
                        / fold_weights[is_label].sum()
                    )
                )
            score = float(np.mean(recalls))
        f1s: list[float] = []
        for label in np.unique(np.concatenate((fold_gold, fold_predicted))):
            true_label = fold_gold == label
            predicted_label = fold_predicted == label
            tp = float(fold_weights[true_label & predicted_label].sum())
            fp = float(fold_weights[~true_label & predicted_label].sum())
            fn = float(fold_weights[true_label & ~predicted_label].sum())
            denominator = 2.0 * tp + fp + fn
            f1s.append(2.0 * tp / denominator if denominator else 0.0)
        fold_scores.append(score)
        fold_f1s.append(float(np.mean(f1s)))
    return float(np.mean(fold_scores)), float(np.mean(fold_f1s))


def bootstrap_classification(
    entries: Sequence[EvaluationEntry],
    *,
    replicates: int,
    bootstrap_seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    systems, seeds, unit_ids, data = _aligned_records(
        entries, "item_id", ("speaker_id", "fold", "gold")
    )
    score_names = {entry.classification_metric or "accuracy" for entry in entries}
    if len(score_names) != 1:
        raise ValueError("classification entries disagree on their evaluator metric")
    score_name = next(iter(score_names))
    if score_name not in {"accuracy", "balanced_accuracy"}:
        raise ValueError(f"unsupported classification score: {score_name}")

    first_rows = [data[systems[0]][seeds[0]][unit_id] for unit_id in unit_ids]
    for row in first_rows:
        if (
            row.get("speaker_id") is None
            or row.get("fold") is None
            or row.get("gold") is None
        ):
            raise ValueError(
                "classification rows need speaker_id, fold, gold, and prediction fields"
            )
    speakers = sorted({str(row["speaker_id"]) for row in first_rows})
    speaker_index = {speaker: index for index, speaker in enumerate(speakers)}
    speaker_codes = np.asarray(
        [speaker_index[str(row["speaker_id"])] for row in first_rows],
        dtype=np.int64,
    )
    gold = np.asarray([str(row["gold"]) for row in first_rows])
    folds = np.asarray([str(row["fold"]) for row in first_rows])
    predictions = np.asarray(
        [
            [
                [str(data[system][seed][unit_id]["prediction"]) for unit_id in unit_ids]
                for seed in seeds
            ]
            for system in systems
        ]
    )

    unit_weights = np.ones(len(unit_ids), dtype=np.float64)
    point_arrays = np.empty((len(systems), len(seeds), 2), dtype=np.float64)
    for system_index in range(len(systems)):
        for seed_index in range(len(seeds)):
            point_arrays[system_index, seed_index] = _fold_classification_metrics(
                gold,
                predictions[system_index, seed_index],
                folds,
                unit_weights,
                score_name,
            )

    rng = np.random.default_rng(bootstrap_seed)
    boot_arrays = np.empty((2, replicates, len(systems)), dtype=np.float64)
    accepted = attempts = 0
    max_attempts = max(100, replicates * 100)
    while accepted < replicates and attempts < max_attempts:
        attempts += 1
        sampled = rng.integers(0, len(speakers), size=len(speakers))
        multiplicity = np.bincount(sampled, minlength=len(speakers))
        weights = multiplicity[speaker_codes].astype(np.float64)
        if any(not np.any((folds == fold) & (weights > 0)) for fold in np.unique(folds)):
            continue
        for system_index in range(len(systems)):
            seed_metrics = [
                _fold_classification_metrics(
                    gold,
                    predictions[system_index, seed_index],
                    folds,
                    weights,
                    score_name,
                )
                for seed_index in range(len(seeds))
            ]
            boot_arrays[:, accepted, system_index] = np.mean(
                np.asarray(seed_metrics, dtype=np.float64), axis=0
            )
        accepted += 1
    if accepted != replicates:
        raise ValueError("could not draw enough speaker samples preserving every fold")

    result: dict[str, Any] = {
        "bootstrap_unit": "speaker",
        "valid_draws": accepted,
        "draw_attempts": attempts,
        "fold_aggregation": "unweighted_mean",
        "metrics": {},
    }
    for metric_index, metric_name in enumerate((score_name, "macro_f1")):
        point = {
            system: {
                seed: float(point_arrays[system_index, seed_index, metric_index])
                for seed_index, seed in enumerate(seeds)
            }
            for system_index, system in enumerate(systems)
        }
        boot = {
            system: boot_arrays[metric_index, :, system_index].tolist()
            for system_index, system in enumerate(systems)
        }
        result["metrics"][metric_name] = _summary(
            systems, seeds, point, boot, confidence
        )
    return result


def verification_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    weights: Sequence[float] | None = None,
    *,
    p_target: float = 0.01,
) -> dict[str, float]:
    """Compute EER and normalized minDCF, respecting tied score thresholds."""
    result = _prepare_verification_evaluator(scores, labels, p_target)(weights)
    return {name: float(np.asarray(value).item()) for name, value in result.items()}


def _prepare_verification_evaluator(
    scores: Sequence[float], labels: Sequence[int], p_target: float
):
    """Pre-sort static scores and return a vectorized weighted ROC evaluator."""
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int8)
    if len(score_array) == 0 or len(score_array) != len(label_array):
        raise ValueError("verification scores and labels must align")
    if not 0.0 < p_target < 1.0 or set(np.unique(label_array)) != {0, 1}:
        raise ValueError("invalid verification prior or labels")
    order = np.argsort(-score_array, kind="mergesort")
    sorted_scores = score_array[order]
    sorted_labels = label_array[order]
    threshold_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )

    def evaluate(
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, float | np.ndarray]:
        weight_array = (
            np.ones((1, len(score_array)), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        scalar = weight_array.ndim == 1
        if scalar:
            weight_array = weight_array[None, :]
        if (
            weight_array.ndim != 2
            or weight_array.shape[1] != len(score_array)
            or np.any(weight_array < 0)
        ):
            raise ValueError("verification weights must align and be non-negative")
        sorted_weights = weight_array[:, order]
        target_total = sorted_weights[:, sorted_labels == 1].sum(axis=1)
        non_target_total = sorted_weights[:, sorted_labels == 0].sum(axis=1)
        if np.any(target_total <= 0.0) or np.any(non_target_total <= 0.0):
            raise ValueError("verification resample needs target and non-target trials")
        target_cumulative = np.cumsum(
            sorted_weights * (sorted_labels == 1)[None, :], axis=1
        )
        non_target_cumulative = np.cumsum(
            sorted_weights * (sorted_labels == 0)[None, :], axis=1
        )
        tpr = target_cumulative[:, threshold_ends] / target_total[:, None]
        fpr = non_target_cumulative[:, threshold_ends] / non_target_total[:, None]
        fnr = 1.0 - tpr
        differences = np.abs(fnr - fpr)
        indices = np.argmin(differences, axis=1)
        rows = np.arange(len(weight_array))
        eer = (fnr[rows, indices] + fpr[rows, indices]) / 2.0
        # The threshold above the maximum score has FNR=1/FPR=0. Include it
        # explicitly, preserving the scalar evaluator's first-minimum tie rule.
        use_initial = np.min(differences, axis=1) >= 1.0
        eer = np.where(use_initial, 0.5, eer)
        costs = p_target * fnr + (1.0 - p_target) * fpr
        normalization = min(p_target, 1.0 - p_target)
        min_dcf = np.minimum(
            p_target / normalization,
            np.min(costs, axis=1) / normalization,
        )
        if scalar or weights is None:
            return {"eer": float(eer[0]), "min_dcf": float(min_dcf[0])}
        return {"eer": eer, "min_dcf": min_dcf}

    return evaluate


def _verification_row_fields(row: Mapping[str, Any]) -> tuple[float, int]:
    score = row.get("score")
    raw_label = row.get("label", row.get("target"))
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError("verification rows need a numeric score")
    if isinstance(raw_label, bool):
        label = int(raw_label)
    elif isinstance(raw_label, (int, float)) and raw_label in (0, 1):
        label = int(raw_label)
    else:
        raise ValueError("verification rows need a binary label/target")
    return float(score), label


def _verification_arrays_from_entry(entry: EvaluationEntry) -> VerificationArrays:
    if entry.verification_arrays is not None:
        return entry.verification_arrays
    rows = _index_records(entry.records, "trial_id")
    trial_ids = sorted(rows)
    labels: list[int] = []
    scores: list[float] = []
    for trial_id in trial_ids:
        score, label = _verification_row_fields(rows[trial_id])
        scores.append(score)
        labels.append(label)
    unit = entry.bootstrap_unit or "speaker"
    if unit == "trial":
        return VerificationArrays(
            trial_ids=np.asarray(trial_ids),
            labels=np.asarray(labels, dtype=np.int8),
            scores=np.asarray(scores, dtype=np.float64),
            bootstrap_unit=unit,
        )
    if unit != "speaker":
        raise ValueError("verification bootstrap_unit must be speaker or trial")
    speaker_names = sorted({
        str(rows[trial_id][key])
        for trial_id in trial_ids
        for key in ("enroll_speaker_id", "test_speaker_id")
    })
    vocabulary = np.asarray(speaker_names)
    speaker_index = {speaker: index for index, speaker in enumerate(speaker_names)}
    return VerificationArrays(
        trial_ids=np.asarray(trial_ids),
        labels=np.asarray(labels, dtype=np.int8),
        scores=np.asarray(scores, dtype=np.float64),
        bootstrap_unit=unit,
        speaker_a=np.asarray(
            [speaker_index[str(rows[trial_id]["enroll_speaker_id"])] for trial_id in trial_ids],
            dtype=np.int32,
        ),
        speaker_b=np.asarray(
            [speaker_index[str(rows[trial_id]["test_speaker_id"])] for trial_id in trial_ids],
            dtype=np.int32,
        ),
        speaker_vocabulary=vocabulary,
    )


def bootstrap_verification(
    entries: Sequence[EvaluationEntry],
    *,
    replicates: int,
    bootstrap_seed: int,
    confidence: float = 0.95,
    p_target: float = 0.01,
) -> dict[str, Any]:
    systems, seeds, _ = _organized(entries)
    entry_grid: dict[str, dict[str, EvaluationEntry]] = {system: {} for system in systems}
    for entry in entries:
        entry_grid[entry.system][entry.seed] = entry
    arrays = {
        system: {
            seed: _verification_arrays_from_entry(entry_grid[system][seed])
            for seed in seeds
        }
        for system in systems
    }
    canonical = arrays[systems[0]][seeds[0]]
    bootstrap_unit = canonical.bootstrap_unit
    if bootstrap_unit not in {"speaker", "trial"}:
        raise ValueError("verification bootstrap_unit must be speaker or trial")
    if len(canonical.trial_ids) == 0 or set(np.unique(canonical.labels)) != {0, 1}:
        raise ValueError("verification needs non-empty target and non-target trials")
    for system in systems:
        for seed in seeds:
            item = arrays[system][seed]
            if item.bootstrap_unit != bootstrap_unit:
                raise ValueError("verification entries disagree on bootstrap_unit")
            if not np.array_equal(item.trial_ids, canonical.trial_ids):
                raise ValueError("verification trial IDs differ across systems/seeds")
            if not np.array_equal(item.labels, canonical.labels):
                raise ValueError("verification labels differ across systems/seeds")
            for field in (
                "trial_left",
                "trial_right",
                "utterance_ids",
                "utterance_speaker",
            ):
                if not np.array_equal(
                    getattr(item, field), getattr(canonical, field)
                ):
                    raise ValueError(
                        f"verification {field} differ across systems/seeds"
                    )
            if item.split_hash != canonical.split_hash:
                raise ValueError("verification split hashes differ across systems/seeds")
            if not np.all(np.isfinite(item.scores)):
                raise ValueError("verification scores must be finite")
            if bootstrap_unit == "speaker" and not (
                np.array_equal(item.speaker_a, canonical.speaker_a)
                and np.array_equal(item.speaker_b, canonical.speaker_b)
                and np.array_equal(item.speaker_vocabulary, canonical.speaker_vocabulary)
            ):
                raise ValueError("verification speaker groups differ across systems/seeds")

    labels = canonical.labels.astype(np.int8, copy=False)
    num_trials = len(labels)
    evaluators: dict[str, dict[str, Any]] = {}
    point_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for system in systems:
        evaluators[system] = {}
        point_metrics[system] = {}
        for seed in seeds:
            evaluator = _prepare_verification_evaluator(
                arrays[system][seed].scores, labels, p_target
            )
            evaluators[system][seed] = evaluator
            point_metrics[system][seed] = evaluator()  # type: ignore[assignment]

    if bootstrap_unit == "speaker":
        if (
            canonical.speaker_a is None
            or canonical.speaker_b is None
            or canonical.speaker_vocabulary is None
            or len(canonical.speaker_vocabulary) < 2
        ):
            raise ValueError("speaker bootstrap needs endpoint speaker arrays")
        speaker_a = canonical.speaker_a.astype(np.int64, copy=False)
        speaker_b = canonical.speaker_b.astype(np.int64, copy=False)
        if np.any((labels == 1) & (speaker_a != speaker_b)):
            raise ValueError("target verification trials must stay within speaker")
        if np.any((labels == 0) & (speaker_a == speaker_b)):
            raise ValueError("non-target verification trials must cross speakers")
        num_speakers = len(canonical.speaker_vocabulary)
    else:
        speaker_a = speaker_b = None
        num_speakers = 0

    # Keep generated weights plus sorted cumulative work bounded. Evaluators
    # process this common draw matrix for every system/seed before it is dropped.
    max_weight_elements = 1_000_000
    chunk_rows = max(1, min(128, max_weight_elements // num_trials))
    boot_arrays = np.empty((2, replicates, len(systems)), dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    accepted = attempts = 0
    max_attempts = max(100, replicates * 100)
    while accepted < replicates and attempts < max_attempts:
        candidate_rows = min(chunk_rows, replicates - accepted)
        if bootstrap_unit == "trial":
            probabilities = np.full(num_trials, 1.0 / num_trials)
            weights = rng.multinomial(
                num_trials, probabilities, size=candidate_rows
            ).astype(np.float64)
        else:
            probabilities = np.full(num_speakers, 1.0 / num_speakers)
            multiplicities = rng.multinomial(
                num_speakers, probabilities, size=candidate_rows
            )
            left = multiplicities[:, speaker_a]
            right = multiplicities[:, speaker_b]
            weights = np.where(labels[None, :] == 1, left, left * right).astype(
                np.float64
            )
        attempts += candidate_rows
        valid = (
            weights[:, labels == 1].sum(axis=1) > 0
        ) & (
            weights[:, labels == 0].sum(axis=1) > 0
        )
        weights = weights[valid]
        if len(weights) == 0:
            continue
        weights = weights[: replicates - accepted]
        count = len(weights)
        for system_index, system in enumerate(systems):
            per_seed = [evaluators[system][seed](weights) for seed in seeds]
            for metric_index, metric_name in enumerate(("eer", "min_dcf")):
                boot_arrays[
                    metric_index, accepted : accepted + count, system_index
                ] = np.mean(
                    np.stack(
                        [np.asarray(item[metric_name]) for item in per_seed], axis=0
                    ),
                    axis=0,
                )
        accepted += count
    if accepted != replicates:
        raise ValueError("could not draw enough valid verification bootstrap replicates")

    result: dict[str, Any] = {
        "bootstrap_unit": bootstrap_unit,
        "valid_draws": accepted,
        "draw_attempts": attempts,
        "weight_chunk_rows": chunk_rows,
        "metrics": {},
    }
    for metric_index, metric_name in enumerate(("eer", "min_dcf")):
        point = {
            system: {
                seed: float(point_metrics[system][seed][metric_name]) for seed in seeds
            }
            for system in systems
        }
        boot = {
            system: boot_arrays[metric_index, :, system_index].tolist()
            for system_index, system in enumerate(systems)
        }
        result["metrics"][metric_name] = _summary(
            systems, seeds, point, boot, confidence
        )
    return result


def _read_records(path: pathlib.Path) -> list[dict[str, Any]]:
    if path.suffix == ".npz":
        raise ValueError(
            "NPZ trial artifacts require their summary descriptor so the score set is unambiguous"
        )
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        for key in ("records", "per_item", "predictions", "trials"):
            if isinstance(payload.get(key), list):
                return [dict(row) for row in payload[key]]
    raise ValueError(f"artifact does not contain a record list: {path}")


def _artifact_records(summary_path: pathlib.Path, value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list):
        return [dict(row) for row in value]
    if isinstance(value, dict):
        if isinstance(value.get("records"), list):
            return [dict(row) for row in value["records"]]
        value = value.get("path")
    if isinstance(value, str):
        path = pathlib.Path(value)
        if not path.is_absolute():
            path = summary_path.parent / path
        return _read_records(path)
    return None


def _npz_trial_arrays(
    summary_path: pathlib.Path, descriptor: Mapping[str, Any]
) -> VerificationArrays:
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("NPZ artifact descriptor needs a path")
    path = pathlib.Path(raw_path)
    if not path.is_absolute():
        path = summary_path.parent / path
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"].item()))
        score_sets = metadata.get("score_sets", descriptor.get("score_sets"))
        if not isinstance(score_sets, list) or len(score_sets) != 1:
            raise ValueError(
                "one verification summary entry must describe exactly one model/pool score set"
            )
        score_key = score_sets[0].get("key")
        if not isinstance(score_key, str) or score_key not in artifact:
            raise ValueError("verification score-set metadata is invalid")
        trial_ids = artifact["trial_id"]
        labels = artifact["label"]
        speaker_vocabulary = artifact["speaker_vocabulary"]
        utterance_ids = artifact["utterance_ids"]
        utterance_speaker = artifact["utterance_speaker"]
        trial_left = artifact["trial_left"]
        trial_right = artifact["trial_right"]
        speaker_a = artifact["speaker_a"]
        speaker_b = artifact["speaker_b"]
        scores = artifact[score_key]
        count = len(trial_ids)
        if any(
            len(values) != count
            for values in (
                labels,
                trial_left,
                trial_right,
                speaker_a,
                speaker_b,
                scores,
            )
        ):
            raise ValueError("verification NPZ arrays have inconsistent lengths")
        if len(utterance_ids) != len(utterance_speaker):
            raise ValueError("verification NPZ utterance arrays have inconsistent lengths")
        result = VerificationArrays(
            trial_ids=np.asarray(trial_ids).copy(),
            labels=np.asarray(labels, dtype=np.int8).copy(),
            scores=np.asarray(scores, dtype=np.float64).copy(),
            bootstrap_unit=str(metadata.get("bootstrap_unit", "speaker")),
            speaker_a=np.asarray(speaker_a, dtype=np.int32).copy(),
            speaker_b=np.asarray(speaker_b, dtype=np.int32).copy(),
            speaker_vocabulary=np.asarray(speaker_vocabulary).copy(),
            trial_left=np.asarray(trial_left, dtype=np.int32).copy(),
            trial_right=np.asarray(trial_right, dtype=np.int32).copy(),
            utterance_ids=np.asarray(utterance_ids).copy(),
            utterance_speaker=np.asarray(utterance_speaker, dtype=np.int32).copy(),
            split_hash=(
                str(metadata["split_hash"])
                if metadata.get("split_hash") is not None
                else None
            ),
        )
    return result


def load_entry(
    task: str, system: str, seed: str, summary_path: pathlib.Path
) -> EvaluationEntry:
    kind = _task_kind(task)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be a JSON object: {summary_path}")
    containers = [summary]
    if kind == "asr" and isinstance(summary.get("dev"), dict):
        containers.insert(0, summary["dev"])
    if kind == "reconstruction":
        keys = ("per_item", "per_item_artifact", "items_artifact", "artifact")
    elif kind in {"asr", "classification"}:
        keys = (
            "predictions", "predictions_descriptor", "predictions_artifact", "artifact"
        )
    else:
        keys = ("trials", "trials_artifact", "predictions_artifact", "artifact")
    records: list[dict[str, Any]] | None = None
    bootstrap_unit = summary.get("bootstrap_unit")
    verification_arrays: VerificationArrays | None = None
    for container in containers:
        for key in keys:
            descriptor = container.get(key)
            if (
                kind == "verification"
                and isinstance(descriptor, Mapping)
                and str(descriptor.get("format", "")).lower() == "npz"
            ):
                verification_arrays = _npz_trial_arrays(summary_path, descriptor)
                bootstrap_unit = verification_arrays.bootstrap_unit
                records = []
            else:
                records = _artifact_records(summary_path, descriptor)
            if records is not None:
                break
        if records is not None:
            break
    if records is None and kind == "classification":
        results = summary.get("results")
        if isinstance(results, Mapping) and len(results) == 1:
            model_result = next(iter(results.values()))
            if isinstance(model_result, Mapping):
                for key in ("predictions", "oof_predictions", "test_predictions"):
                    candidate = model_result.get(key)
                    if isinstance(candidate, list):
                        records = [dict(row) for row in candidate]
                        break
    if records is None:
        raise ValueError(f"no standardized {task} records/artifact in {summary_path}")
    if bootstrap_unit is not None and not isinstance(bootstrap_unit, str):
        raise ValueError("bootstrap_unit must be a string")
    classification_metric: str | None = None
    if kind == "classification":
        results = summary.get("results")
        if isinstance(results, Mapping) and len(results) == 1:
            model_result = next(iter(results.values()))
            if isinstance(model_result, Mapping):
                classification_metric = (
                    "balanced_accuracy"
                    if "balanced_accuracy" in model_result
                    else "accuracy"
                )
        if classification_metric is None:
            classification_metric = (
                "balanced_accuracy"
                if "balanced_accuracy" in summary
                else "accuracy"
            )
    return EvaluationEntry(
        task,
        system,
        seed,
        records,
        bootstrap_unit,
        classification_metric,
        verification_arrays,
    )


def analyze_entries(
    entries: Sequence[EvaluationEntry],
    *,
    replicates: int,
    bootstrap_seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    result: dict[str, Any] = {
        "schema_version": "tacl-paired-bootstrap-v1",
        "bootstrap_seed": bootstrap_seed,
        "replicates": replicates,
        "confidence": confidence,
        "tasks": {},
    }
    grouped: dict[str, list[EvaluationEntry]] = {}
    for entry in entries:
        _task_kind(entry.task)
        grouped.setdefault(entry.task, []).append(entry)
    dispatch = {
        "reconstruction": bootstrap_reconstruction,
        "asr": bootstrap_asr,
        "classification": bootstrap_classification,
        "verification": bootstrap_verification,
    }
    for task in sorted(grouped):
        result["tasks"][task] = dispatch[_task_kind(task)](
            grouped[task],
            replicates=replicates,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paired TACL bootstrap intervals from saved evaluator artifacts.",
        epilog=(
            "Repeat --entry TASK SYSTEM SEED SUMMARY. TASK is reconstruction, asr, "
            "classification, or verification and may have a named suffix such as "
            "classification:emotion; use SEED=none for deterministic tasks."
        ),
    )
    parser.add_argument(
        "--entry",
        action="append",
        nargs=4,
        metavar=("TASK", "SYSTEM", "SEED", "SUMMARY"),
        required=True,
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be strictly between zero and one")

    entries: list[EvaluationEntry] = []
    for task, system, seed, raw_path in args.entry:
        try:
            _task_kind(task)
        except ValueError as exc:
            parser.error(str(exc))
        summary_path = pathlib.Path(raw_path)
        entries.append(load_entry(task, system, seed, summary_path))
    result = analyze_entries(
        entries,
        replicates=args.replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
