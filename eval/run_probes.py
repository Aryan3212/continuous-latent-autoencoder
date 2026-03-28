from __future__ import annotations

import json
import pathlib
import subprocess
import time
from typing import Any, Dict, Optional


def run_all_probes(
    *,
    run_dir: str,
    step: int,
    exp_cfg: Dict[str, Any],
    ckpt_path: str,
    python_bin: str = "python",
) -> Dict[str, Any]:
    """
    Frozen-encoder probes:
      - ASR: train a small CTC head, report WER
      - Emotion: pooled embedding + MLP, report macro-F1/acc
      - Gender: pooled embedding + MLP, report acc
    """
    out_dir = pathlib.Path(run_dir) / "eval" / f"step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    def _run(name: str, cmd: list[str]) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [Eval Step {step}] Starting {name}...", flush=True)
        start_t = time.perf_counter()
        try:
            # We use subprocess.run to wait for the child process.
            # Output is piped to stdout so it's visible in the main log.
            subprocess.run(cmd, check=True)
            elapsed = time.perf_counter() - start_t
            print(f"[{time.strftime('%H:%M:%S')}] [Eval Step {step}] {name} finished in {elapsed:.2f}s", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"[{time.strftime('%H:%M:%S')}] [Eval Step {step}] {name} FAILED with exit code {e.returncode}", flush=True)
            raise

    eval_cfg = exp_cfg.get("eval") or {}
    config_path = exp_cfg.get("_resolved_config_path")
    if not config_path:
        raise ValueError("exp_cfg must include _resolved_config_path")

    # Emotion
    emo = (eval_cfg.get("emotion") or {}) if isinstance(eval_cfg.get("emotion"), dict) else {}
    if emo.get("enabled", False):
        out = out_dir / "emotion.json"
        _run(
            "Emotion Probe",
            [
                python_bin,
                "-m",
                "eval.eval_emotion",
                "--config",
                config_path,
                "--ckpt",
                ckpt_path,
                "--train_manifest",
                emo["train_manifest"],
                "--dev_manifest",
                emo["dev_manifest"],
                "--label_key",
                emo.get("label_key", "emotion"),
                "--steps",
                str(int(emo.get("steps", 2000))),
                "--batch_size",
                str(int(emo.get("batch_size", 64))),
                "--out",
                str(out),
            ]
        )
        results["emotion"] = json.loads(out.read_text())

    # Gender
    gen = (eval_cfg.get("gender") or {}) if isinstance(eval_cfg.get("gender"), dict) else {}
    if gen.get("enabled", False):
        out = out_dir / "gender.json"
        _run(
            "Gender Probe",
            [
                python_bin,
                "-m",
                "eval.eval_gender",
                "--config",
                config_path,
                "--ckpt",
                ckpt_path,
                "--train_manifest",
                gen["train_manifest"],
                "--dev_manifest",
                gen["dev_manifest"],
                "--label_key",
                gen.get("label_key", "gender"),
                "--steps",
                str(int(gen.get("steps", 1500))),
                "--batch_size",
                str(int(gen.get("batch_size", 64))),
                "--out",
                str(out),
            ]
        )
        results["gender"] = json.loads(out.read_text())

    # ASR
    asr = (eval_cfg.get("asr") or {}) if isinstance(eval_cfg.get("asr"), dict) else {}
    if asr.get("enabled", False):
        out = out_dir / "asr.json"
        cmd = [
            python_bin,
            "-m",
            "eval.eval_asr",
            "--config",
            config_path,
            "--ckpt",
            ckpt_path,
            "--train_manifest",
            asr["train_manifest"],
            "--dev_manifest",
            asr["dev_manifest"],
            "--text_key",
            asr.get("text_key", "text"),
            "--steps",
            str(int(asr.get("steps", 8000))),
            "--batch_size",
            str(int(asr.get("batch_size", 16))),
            "--out",
            str(out),
        ]
        if asr.get("use_latent", False):
            cmd.append("--use_latent")
        _run("ASR Probe", cmd)
        results["asr"] = json.loads(out.read_text())

    # Latent Visualization (PCA/UMAP)
    vis_out = out_dir / "latents.png"
    vis_manifest = eval_cfg.get("vis_manifest") or exp_cfg["data"].get("val_manifest") or exp_cfg["data"]["train_manifest"]
    
    try:
        _run(
            "Latent Visualization",
            [
                python_bin,
                "scripts/visualize_latents.py",
                "--config",
                config_path,
                "--ckpt",
                ckpt_path,
                "--manifest",
                vis_manifest,
                "--out",
                str(vis_out),
                "--limit",
                "200" # Reduced from 500 for faster evaluation
            ]
        )
        results["visualization"] = str(vis_out)
    except Exception as e:
        print(f"Visualization failed: {e}")

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    (pathlib.Path(run_dir) / f"eval_step_{step}.json").write_text(json.dumps(results, indent=2))
    return results
