from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np

from eval.bootstrap_statistics import (
    EvaluationEntry,
    _asr_error_counts,
    _asr_rate,
    _fold_classification_metrics,
    bootstrap_asr,
    bootstrap_classification,
    bootstrap_reconstruction,
    bootstrap_verification,
    load_entry,
    sample_standard_deviation,
    verification_metrics,
)


class BootstrapStatisticsTest(unittest.TestCase):
    def test_reconstruction_keeps_draws_paired_and_separates_seed_sd(self) -> None:
        entries = []
        for system, offset in (("a", 0.0), ("b", 1.0)):
            for seed in ("0", "1", "2"):
                seed_offset = float(seed)
                entries.append(
                    EvaluationEntry(
                        "reconstruction",
                        system,
                        seed,
                        [
                            {
                                "item_id": f"u{index}",
                                "metrics": {"loss": value + seed_offset + offset},
                            }
                            for index, value in enumerate((1.0, 2.0, 4.0))
                        ],
                    )
                )

        result = bootstrap_reconstruction(
            entries, replicates=100, bootstrap_seed=7
        )["metrics"]["loss"]
        self.assertAlmostEqual(result["systems"]["a"]["seed_mean"], 10.0 / 3.0)
        self.assertAlmostEqual(result["systems"]["a"]["seed_sample_sd"], 1.0)
        paired = result["paired_differences"]["b - a"]
        self.assertAlmostEqual(paired["estimate_b_minus_a"], 1.0)
        self.assertAlmostEqual(paired["bootstrap_ci"]["low"], 1.0)
        self.assertAlmostEqual(paired["bootstrap_ci"]["high"], 1.0)

    def test_asr_recomputes_corpus_metrics_from_joint_utterance_draws(self) -> None:
        rows_a = [
            {"item_id": "u1", "reference": "a b", "hypothesis": "a b"},
            {"item_id": "u2", "reference": "c", "hypothesis": "x"},
        ]
        rows_b = [
            {"item_id": "u1", "reference": "a b", "hypothesis": "a"},
            {"item_id": "u2", "reference": "c", "hypothesis": "c"},
        ]
        result = bootstrap_asr(
            [
                EvaluationEntry("asr", "a", "0", rows_a),
                EvaluationEntry("asr", "b", "0", rows_b),
            ],
            replicates=50,
            bootstrap_seed=3,
        )
        self.assertAlmostEqual(result["metrics"]["wer"]["systems"]["a"]["seed_mean"], 1 / 3)
        self.assertIsNone(
            result["metrics"]["wer"]["systems"]["a"]["seed_sample_sd"]
        )

    def test_additive_asr_counts_match_jiwer_corpus_rates(self) -> None:
        from jiwer import cer, wer

        references = ["a b", "c d e", "hello"]
        hypotheses = ["a x", "c e", "hallo extra"]
        for name, expected in (
            ("wer", wer(references, hypotheses)),
            ("cer", cer(references, hypotheses)),
        ):
            counts = [
                _asr_error_counts(reference, hypothesis, name)
                for reference, hypothesis in zip(references, hypotheses)
            ]
            self.assertAlmostEqual(_asr_rate(counts), expected)

    def test_classification_rejects_changed_gold_or_speaker_group(self) -> None:
        common = [
            {
                "item_id": "u1",
                "speaker_id": "s1",
                "fold": 0,
                "gold": "x",
                "prediction": "x",
            },
            {
                "item_id": "u2",
                "speaker_id": "s2",
                "fold": 0,
                "gold": "y",
                "prediction": "y",
            },
        ]
        changed = [dict(row) for row in common]
        changed[0]["speaker_id"] = "different"
        with self.assertRaisesRegex(ValueError, "invariant fields"):
            bootstrap_classification(
                [
                    EvaluationEntry("classification", "a", "0", common),
                    EvaluationEntry("classification", "b", "0", changed),
                ],
                replicates=10,
                bootstrap_seed=0,
            )

    def test_classification_bootstraps_whole_speaker_clusters(self) -> None:
        rows = [
            {
                "item_id": "u1",
                "speaker_id": "s1",
                "fold": 0,
                "gold": "x",
                "prediction": "x",
            },
            {
                "item_id": "u2",
                "speaker_id": "s1",
                "fold": 0,
                "gold": "y",
                "prediction": "y",
            },
            {
                "item_id": "u3",
                "speaker_id": "s2",
                "fold": 0,
                "gold": "x",
                "prediction": "x",
            },
            {
                "item_id": "u4",
                "speaker_id": "s2",
                "fold": 0,
                "gold": "y",
                "prediction": "y",
            },
        ]
        result = bootstrap_classification(
            [
                EvaluationEntry("classification", "a", "0", rows),
                EvaluationEntry("classification", "b", "0", rows),
            ],
            replicates=25,
            bootstrap_seed=11,
        )
        self.assertEqual(result["bootstrap_unit"], "speaker")
        self.assertEqual(
            result["metrics"]["macro_f1"]["systems"]["a"]["bootstrap_ci"],
            {"low": 1.0, "high": 1.0},
        )

    def test_classification_matches_unweighted_mean_of_fold_metrics(self) -> None:
        score, _ = _fold_classification_metrics(
            gold=np.asarray(["x", "x", "x", "x"]),
            predicted=np.asarray(["x", "y", "y", "y"]),
            folds=np.asarray([0, 1, 1, 1]),
            weights=np.ones(4),
            score_name="accuracy",
        )
        self.assertEqual(score, 0.5)

    def test_classification_supports_balanced_accuracy(self) -> None:
        score, macro_f1 = _fold_classification_metrics(
            gold=np.asarray(["x", "x", "x", "y"]),
            predicted=np.asarray(["x", "x", "x", "x"]),
            folds=np.zeros(4),
            weights=np.ones(4),
            score_name="balanced_accuracy",
        )
        self.assertEqual(score, 0.5)
        self.assertAlmostEqual(macro_f1, 3.0 / 7.0)

    def test_classification_rejects_a_resample_that_empties_a_fold(self) -> None:
        with self.assertRaisesRegex(ValueError, "emptied required fold"):
            _fold_classification_metrics(
                gold=np.asarray(["x", "y"]),
                predicted=np.asarray(["x", "y"]),
                folds=np.asarray([0, 1]),
                weights=np.asarray([1.0, 0.0]),
                score_name="accuracy",
            )

    def test_verification_speaker_bootstrap_uses_shared_weighted_trials(self) -> None:
        identities = [
            ("t1", 1, "s1", "s1"),
            ("t2", 1, "s2", "s2"),
            ("t3", 1, "s3", "s3"),
            ("n12", 0, "s1", "s2"),
            ("n13", 0, "s1", "s3"),
            ("n23", 0, "s2", "s3"),
        ]
        good = [
            {
                "trial_id": trial_id,
                "label": label,
                "enroll_speaker_id": left,
                "test_speaker_id": right,
                "score": 0.9 if label else 0.1,
            }
            for trial_id, label, left, right in identities
        ]
        bad = [dict(row, score=1.0 - row["score"]) for row in good]
        result = bootstrap_verification(
            [
                EvaluationEntry("verification", "good", "none", good, "speaker"),
                EvaluationEntry("verification", "bad", "none", bad, "speaker"),
            ],
            replicates=40,
            bootstrap_seed=5,
        )
        self.assertEqual(result["bootstrap_unit"], "speaker")
        self.assertEqual(result["valid_draws"], 40)
        self.assertAlmostEqual(
            result["metrics"]["eer"]["systems"]["good"]["seed_mean"], 0.0
        )
        self.assertGreaterEqual(
            result["metrics"]["eer"]["systems"]["bad"]["seed_mean"], 0.5
        )

    def test_verification_metric_respects_trial_weights(self) -> None:
        result = verification_metrics(
            scores=[0.9, 0.8, 0.2, 0.1],
            labels=[1, 1, 0, 0],
            weights=[2.0, 1.0, 3.0, 1.0],
        )
        self.assertEqual(result["eer"], 0.0)
        self.assertEqual(result["min_dcf"], 0.0)

    def test_load_entry_accepts_descriptor_object_and_label_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            artifact = root / "predictions.jsonl"
            artifact.write_text(
                json.dumps(
                    {
                        "item_id": "u1",
                        "speaker_id": "s1",
                        "gold_label": "x",
                        "predicted_label": "x",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps({"predictions": {"path": artifact.name}}),
                encoding="utf-8",
            )
            entry = load_entry("classification:emotion", "ours", "0", summary)
            # Aliases are normalized when records are indexed for analysis.
            self.assertEqual(entry.records[0]["gold_label"], "x")

    def test_load_entry_expands_one_npz_verification_score_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            artifact = root / "trials.npz"
            metadata = {
                "bootstrap_unit": "speaker",
                "split_hash": "fixed-split",
                "score_sets": [
                    {"key": "scores_000", "model": "ours", "pool": "meanstd"}
                ],
            }
            np.savez_compressed(
                artifact,
                trial_id=np.asarray([0, 1]),
                label=np.asarray([1, 0], dtype=np.int8),
                utterance_ids=np.asarray(["u1", "u2"]),
                utterance_speaker=np.asarray([0, 1]),
                speaker_vocabulary=np.asarray(["s1", "s2"]),
                trial_left=np.asarray([0, 0]),
                trial_right=np.asarray([0, 1]),
                speaker_a=np.asarray([0, 0]),
                speaker_b=np.asarray([0, 1]),
                metadata=np.asarray(json.dumps(metadata)),
                scores_000=np.asarray([0.9, 0.1], dtype=np.float32),
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "bootstrap_unit": "speaker",
                        "predictions_artifact": {
                            "path": artifact.name,
                            "format": "npz",
                        },
                    }
                ),
                encoding="utf-8",
            )
            entry = load_entry("verification", "ours", "none", summary)
            self.assertEqual(entry.bootstrap_unit, "speaker")
            self.assertEqual(entry.records, [])
            self.assertIsNotNone(entry.verification_arrays)
            assert entry.verification_arrays is not None
            self.assertEqual(entry.verification_arrays.speaker_a[0], 0)
            self.assertAlmostEqual(entry.verification_arrays.scores[1], 0.1, places=6)

    def test_load_entry_detects_age_balanced_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            summary = root / "age.json"
            summary.write_text(
                json.dumps(
                    {
                        "results": {
                            "ours": {
                                "balanced_accuracy": 0.75,
                                "predictions": [
                                    {
                                        "item_id": "u1",
                                        "speaker_id": "s1",
                                        "fold": 0,
                                        "gold": "20s",
                                        "prediction": "20s",
                                    }
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            entry = load_entry("classification:age", "ours", "0", summary)
            self.assertEqual(entry.classification_metric, "balanced_accuracy")

    def test_sample_sd_requires_repeated_probe_runs(self) -> None:
        self.assertIsNone(sample_standard_deviation([1.0]))
        self.assertAlmostEqual(sample_standard_deviation([1.0, 2.0, 3.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
