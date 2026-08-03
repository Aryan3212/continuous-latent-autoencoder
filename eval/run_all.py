from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict

from config import load_config
from eval.run_probes import run_all_probes
from eval.runner import aggregate_status, read_json_result, run_command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reconstruction metrics and configured frozen-feature probes."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Reconstruction manifest; required unless --skip_recon is set.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output root; each checkpoint is written to step_<N>/ beneath it.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step for output naming; defaults to the checkpoint step.",
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--skip_recon", action="store_true")
    parser.add_argument("--skip_probes", action="store_true")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also create the slower latent visualization (off by default).",
    )
    parser.add_argument("--recon_batch_size", type=int, default=8)
    parser.add_argument("--recon_max_batches", type=int, default=50)
    parser.add_argument("--recon_timeout_seconds", type=int, default=1800)
    parser.add_argument("--probe_timeout_seconds", type=int, default=1800)
    args = parser.parse_args()
    if min(
        args.recon_batch_size,
        args.recon_max_batches,
        args.recon_timeout_seconds,
        args.probe_timeout_seconds,
    ) < 1:
        parser.error("evaluation batch, budget, and timeout values must be positive")
    if not args.skip_recon and not args.manifest:
        parser.error("--manifest is required unless --skip_recon is set")

    cfg = load_config(args.config)
    step = args.step
    if step is None:
        from eval.common import checkpoint_step

        step = checkpoint_step(args.ckpt)
        if step is None:
            parser.error("checkpoint has no embedded step; pass --step explicitly")
        print(f"[run_all] Using checkpoint step {step}.", flush=True)
    if step < 0:
        parser.error("--step must be non-negative")

    output_dir = pathlib.Path(args.out_dir) / f"step_{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"step": int(step), "_status": {}}
    recon_path = output_dir / "recon.json"
    recon_path.unlink(missing_ok=True)
    (output_dir / "summary.json").unlink(missing_ok=True)
    for artifact_name in (
        "emotion.json",
        "gender.json",
        "asr.json",
        "asr.train_filtered.jsonl",
        "asr.dev_filtered.jsonl",
        "latents.png",
    ):
        (output_dir / artifact_name).unlink(missing_ok=True)

    if args.skip_recon:
        results["_status"]["recon"] = {
            "status": "skipped",
            "reason": "disabled by --skip_recon",
        }
    else:
        assert args.manifest is not None
        status = run_command(
            label="Reconstruction evaluation",
            command=[
                args.python_bin,
                "-m",
                "eval.eval_recon",
                "--config",
                args.config,
                "--ckpt",
                args.ckpt,
                "--manifest",
                args.manifest,
                "--batch_size",
                str(args.recon_batch_size),
                "--max_batches",
                str(args.recon_max_batches),
                "--out",
                str(recon_path),
            ],
            step=int(step),
            timeout_seconds=args.recon_timeout_seconds,
        )
        recon = read_json_result(recon_path, status)
        results["_status"]["recon"] = status
        if recon is not None:
            results["recon"] = recon

    if args.skip_probes:
        results["_status"]["probes"] = {
            "status": "skipped",
            "reason": "disabled by --skip_probes",
        }
    elif not cfg.eval.enabled:
        results["_status"]["probes"] = {
            "status": "skipped",
            "reason": "eval.enabled is false",
        }
    else:
        probe_results = run_all_probes(
            output_dir=output_dir,
            step=int(step),
            exp_cfg=cfg,
            config_path=args.config,
            ckpt_path=args.ckpt,
            python_bin=args.python_bin,
            include_visualization=args.visualize,
            timeout_seconds=args.probe_timeout_seconds,
        )
        results["probes"] = probe_results
        child_statuses = list(probe_results.get("_status", {}).values())
        results["_status"]["probes"] = {
            "status": aggregate_status(child_statuses)
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[run_all] Wrote {summary_path}", flush=True)
    top_statuses = results["_status"].values()
    if any(
        status.get("status") in {"failed", "timed_out", "completed_with_errors"}
        for status in top_statuses
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
