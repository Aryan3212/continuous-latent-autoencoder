from __future__ import annotations

import json
import pathlib
import subprocess
import time
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from utils.schema import Config


def run_all_probes(
    *,
    run_dir: str,
    step: int,
    exp_cfg: "Config",
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
    timing: Dict[str, float] = {}

    def _run(name: str, cmd: list[str]) -> bool:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [Eval Step {step}] Starting {name}...", flush=True)
        start_t = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, check=True,
                stderr=subprocess.PIPE,
                timeout=1800,  # 30 min hard timeout per probe
            )
            elapsed = time.perf_counter() - start_t
            timing[name] = elapsed
            print(f"[{time.strftime('%H:%M:%S')}] [Eval Step {step}] {name} finished in {elapsed:.1f}s", flush=True)
            return True
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - start_t
            timing[name] = -elapsed  # negative = failed
            stderr_tail = (e.stderr or b"").decode(errors="replace")[-2000:]
            print(f"[{time.strftime('%H:%M:%S')}] [Eval Step {step}] {name} FAILED "
                  f"(exit {e.returncode}, {elapsed:.1f}s)", flush=True)
            if stderr_tail.strip():
                print(f"  stderr (last 2000 chars):\n{stderr_tail}", flush=True)
            return False
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_t
            timing[name] = -elapsed
            print(f"[{time.strftime('%H:%M:%S')}] [Eval Step {step}] {name} TIMED OUT after {elapsed:.1f}s", flush=True)
            return False

    config_path = exp_cfg.resolved_config_path
    if not config_path:
        raise ValueError("exp_cfg must have resolved_config_path set before passing to run_all_probes")

    # Utterance-level probes (gender / emotion).
    def _run_utt_probe(name: str, pcfg: Any, key: str, hidden: int) -> None:
        if not pcfg.train_manifest or not pcfg.dev_manifest:
            print(f"[Eval Step {step}] {name} probe enabled but eval.{key}.train_manifest/"
                  f"dev_manifest are not set; skipping.", flush=True)
            return
        out = out_dir / f"{key}.json"
        cmd = [
            python_bin,
            "-m",
            "eval.eval_cls_probe",
            "--config",
            config_path,
            "--ckpt",
            ckpt_path,
            "--train_manifest",
            str(pcfg.train_manifest),
            "--dev_manifest",
            str(pcfg.dev_manifest),
            "--label_key",
            str(pcfg.label_key),
            "--steps",
            str(int(pcfg.steps)),
            "--hidden",
            str(hidden),
            "--batch_size",
            str(int(pcfg.batch_size)),
            "--out",
            str(out),
        ]
        seg = pcfg.segment_seconds
        if seg is not None:
            cmd.extend(["--segment_seconds", str(seg)])
        ok = _run(f"{name} Probe", cmd)
        if ok and out.exists():
            results[key] = json.loads(out.read_text())

    if exp_cfg.eval.emotion.enabled:
        _run_utt_probe("Emotion", exp_cfg.eval.emotion, "emotion", 256)

    if exp_cfg.eval.gender.enabled:
        _run_utt_probe("Gender", exp_cfg.eval.gender, "gender", 128)

    # ASR
    if exp_cfg.eval.asr.enabled:
        asr = exp_cfg.eval.asr
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
            asr.train_manifest,
            "--dev_manifest",
            asr.dev_manifest,
            "--text_key",
            asr.text_key,
            "--steps",
            str(asr.steps),
            "--batch_size",
            str(asr.batch_size),
            "--segment_seconds",
            str(asr.segment_seconds),
            "--out",
            str(out),
        ]
        if asr.max_samples:
            cmd.extend(["--max_samples", str(asr.max_samples)])
        ok = _run("ASR Probe", cmd)
        if ok and out.exists():
            results["asr"] = json.loads(out.read_text())

    # Latent Visualization (PCA/UMAP)
    vis_out = out_dir / "latents.png"
    vis_manifest = exp_cfg.data.val_manifest or exp_cfg.data.train_manifest

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
                "200"
            ]
        )
        if vis_out.exists():
            results["visualization"] = str(vis_out)
    except Exception as e:
        print(f"Visualization failed: {e}")

    results["_timing"] = timing
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    (pathlib.Path(run_dir) / f"eval_step_{step}.json").write_text(json.dumps(results, indent=2))
    return results
