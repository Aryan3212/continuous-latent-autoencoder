from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

from eval.runner import read_json_result, run_command

if TYPE_CHECKING:
    from schema import Config


@dataclass(frozen=True)
class ClassificationProbe:
    key: str
    label: str
    config: Any


def run_all_probes(
    *,
    output_dir: pathlib.Path,
    step: int,
    exp_cfg: "Config",
    config_path: str,
    ckpt_path: str,
    python_bin: str,
    include_visualization: bool = False,
    timeout_seconds: int = 1800,
) -> Dict[str, Any]:
    """Run enabled frozen-encoder probes into one exact output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    statuses: Dict[str, Dict[str, Any]] = {}
    for artifact_name in (
        "emotion.json",
        "gender.json",
        "asr.json",
        "asr.train_filtered.jsonl",
        "asr.dev_filtered.jsonl",
        "latents.png",
    ):
        (output_dir / artifact_name).unlink(missing_ok=True)

    if not exp_cfg.eval.enabled:
        return {
            "_status": {
                "eval": {
                    "status": "skipped",
                    "reason": "eval.enabled is false",
                }
            }
        }

    classification_probes = (
        ClassificationProbe("emotion", "Emotion probe", exp_cfg.eval.emotion),
        ClassificationProbe("gender", "Gender probe", exp_cfg.eval.gender),
    )
    for probe in classification_probes:
        cfg = probe.config
        if not cfg.enabled:
            statuses[probe.key] = {"status": "skipped", "reason": "disabled"}
            continue
        if not cfg.train_manifest or not cfg.dev_manifest:
            statuses[probe.key] = {
                "status": "skipped",
                "reason": (
                    f"eval.{probe.key} requires train_manifest and dev_manifest"
                ),
            }
            print(
                f"[Eval step {step}] Skipping {probe.label}: "
                f"configure eval.{probe.key}.train_manifest and dev_manifest.",
                flush=True,
            )
            continue

        output_path = output_dir / f"{probe.key}.json"
        command = [
            python_bin,
            "-m",
            "eval.eval_cls_probe",
            "--config",
            config_path,
            "--ckpt",
            ckpt_path,
            "--train_manifest",
            str(cfg.train_manifest),
            "--dev_manifest",
            str(cfg.dev_manifest),
            "--label_key",
            str(cfg.label_key),
            "--steps",
            str(cfg.steps),
            "--hidden",
            str(cfg.hidden),
            "--batch_size",
            str(cfg.batch_size),
            "--lr",
            str(cfg.lr),
            "--seed",
            str(cfg.seed),
            "--out",
            str(output_path),
        ]
        if cfg.segment_seconds is not None:
            command.extend(["--segment_seconds", str(cfg.segment_seconds)])
        status = run_command(
            label=probe.label,
            command=command,
            step=step,
            timeout_seconds=timeout_seconds,
        )
        statuses[probe.key] = status
        result = read_json_result(output_path, status)
        if result is not None:
            results[probe.key] = result

    asr = exp_cfg.eval.asr
    if not asr.enabled:
        statuses["asr"] = {"status": "skipped", "reason": "disabled"}
    elif not asr.train_manifest or not asr.dev_manifest:
        statuses["asr"] = {
            "status": "skipped",
            "reason": "eval.asr requires train_manifest and dev_manifest",
        }
        print(
            f"[Eval step {step}] Skipping ASR probe: configure "
            "eval.asr.train_manifest and dev_manifest.",
            flush=True,
        )
    else:
        output_path = output_dir / "asr.json"
        command = [
            python_bin,
            "-m",
            "eval.eval_asr",
            "--config",
            config_path,
            "--ckpt",
            ckpt_path,
            "--train_manifest",
            str(asr.train_manifest),
            "--dev_manifest",
            str(asr.dev_manifest),
            "--text_key",
            asr.text_key,
            "--steps",
            str(asr.steps),
            "--batch_size",
            str(asr.batch_size),
            "--seed",
            str(asr.seed),
            "--segment_seconds",
            str(asr.segment_seconds),
            "--out",
            str(output_path),
        ]
        if asr.max_samples:
            command.extend(["--max_samples", str(asr.max_samples)])
        status = run_command(
            label="ASR probe",
            command=command,
            step=step,
            timeout_seconds=timeout_seconds,
        )
        statuses["asr"] = status
        result = read_json_result(output_path, status)
        if result is not None:
            results["asr"] = result

    if include_visualization:
        output_path = output_dir / "latents.png"
        manifest = exp_cfg.data.val_manifest or exp_cfg.data.train_manifest
        status = run_command(
            label="Latent visualization",
            command=[
                python_bin,
                "scripts/visualize_latents.py",
                "--config",
                config_path,
                "--ckpt",
                ckpt_path,
                "--manifest",
                manifest,
                "--out",
                str(output_path),
                "--limit",
                "200",
            ],
            step=step,
            timeout_seconds=timeout_seconds,
        )
        statuses["visualization"] = status
        if status["status"] == "completed" and output_path.exists():
            results["visualization"] = str(output_path)
        elif status["status"] == "completed":
            status.update({
                "status": "failed",
                "reason": f"missing output file: {output_path}",
            })
    else:
        statuses["visualization"] = {
            "status": "skipped",
            "reason": "use --visualize to enable",
        }

    results["_status"] = statuses
    return results
