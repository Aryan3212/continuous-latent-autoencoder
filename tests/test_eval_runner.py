from __future__ import annotations

import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, cast

from eval.run_probes import run_all_probes
from eval.runner import aggregate_status, read_json_result, run_command


class EvalRunnerTest(unittest.TestCase):
    def test_aggregate_status_distinguishes_skips_and_errors(self) -> None:
        self.assertEqual(
            aggregate_status([{"status": "skipped"}, {"status": "skipped"}]),
            "skipped",
        )
        self.assertEqual(
            aggregate_status([{"status": "completed"}, {"status": "skipped"}]),
            "completed",
        )
        self.assertEqual(
            aggregate_status([{"status": "completed"}, {"status": "failed"}]),
            "completed_with_errors",
        )

    def test_invalid_json_converts_completed_status_to_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "result.json"
            path.write_text("not json", encoding="utf-8")
            status = {"status": "completed"}
            self.assertIsNone(read_json_result(path, status))
            self.assertEqual(status["status"], "failed")

    def test_missing_command_is_reported_as_failure(self) -> None:
        status = run_command(
            label="missing",
            command=["/definitely/not/a/real/command"],
            step=1,
        )
        self.assertEqual(status["status"], "failed")
        self.assertIn("reason", status)

    def test_probe_runner_removes_stale_artifacts_when_everything_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            for name in (
                "emotion.json",
                "gender.json",
                "asr.json",
                "asr.train_filtered.jsonl",
                "asr.dev_filtered.jsonl",
                "latents.png",
            ):
                (output_dir / name).write_text("stale", encoding="utf-8")

            disabled = SimpleNamespace(enabled=False)
            config = SimpleNamespace(
                eval=SimpleNamespace(
                    enabled=True,
                    emotion=disabled,
                    gender=disabled,
                    asr=disabled,
                )
            )
            result = run_all_probes(
                output_dir=output_dir,
                step=1,
                exp_cfg=cast(Any, config),
                config_path="unused.yaml",
                ckpt_path="unused.pt",
                python_bin="python",
            )

            self.assertTrue(all(
                status["status"] == "skipped"
                for status in result["_status"].values()
            ))
            self.assertFalse(any(output_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
