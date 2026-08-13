from __future__ import annotations

import argparse
import json
import pathlib
import shutil
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
    parser.add_argument("--recon_audio_limit", type=int, default=20)
    parser.add_argument("--recon_segment_seconds", type=float, default=None)
    parser.add_argument(
        "--report_evals",
        action="store_true",
        help="Run SUBESCO temporal heads and MOS-colored t-SNE/UMAP diagnostics.",
    )
    parser.add_argument(
        "--subesco_dir",
        default=None,
        help="Explicit SUBESCO root; report tasks are recorded as skipped when absent.",
    )
    parser.add_argument("--report_seed", type=int, default=0)
    parser.add_argument("--temporal_max_utts", type=int, default=2100)
    parser.add_argument("--temporal_attn_epochs", type=int, default=40)
    parser.add_argument("--temporal_transformer_epochs", type=int, default=30)
    parser.add_argument("--repr_viz_max_utts", type=int, default=300)
    args = parser.parse_args()
    if min(
        args.recon_batch_size,
        args.recon_max_batches,
        args.recon_timeout_seconds,
        args.probe_timeout_seconds,
        args.temporal_max_utts,
        args.temporal_attn_epochs,
        args.temporal_transformer_epochs,
        args.repr_viz_max_utts,
    ) < 1:
        parser.error("evaluation batch, budget, and timeout values must be positive")
    if args.recon_audio_limit < 0:
        parser.error("--recon_audio_limit must be non-negative")
    if args.recon_segment_seconds is not None and args.recon_segment_seconds <= 0:
        parser.error("--recon_segment_seconds must be positive")
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
    recon_audio_dir = output_dir / "reconstruction_audio"
    if recon_audio_dir.exists():
        shutil.rmtree(recon_audio_dir)
    (output_dir / "summary.json").unlink(missing_ok=True)
    for artifact_name in (
        "emotion.json",
        "gender.json",
        "asr.json",
        "asr.train_filtered.jsonl",
        "asr.dev_filtered.jsonl",
        "latents.png",
        "emotion_temporal_attn.json",
        "emotion_temporal_transformer.json",
        "repr_tsne_umap_mos.png",
        "repr_tsne_umap_mos.json",
    ):
        (output_dir / artifact_name).unlink(missing_ok=True)

    if args.skip_recon:
        results["_status"]["recon"] = {
            "status": "skipped",
            "reason": "disabled by --skip_recon",
        }
    else:
        assert args.manifest is not None
        recon_command = [
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
            "--audio_dir",
            str(recon_audio_dir),
            "--audio_limit",
            str(args.recon_audio_limit),
        ]
        if args.recon_segment_seconds is not None:
            recon_command.extend(
                ["--segment_seconds", str(args.recon_segment_seconds)]
            )
        status = run_command(
            label="Reconstruction evaluation",
            command=recon_command,
            step=int(step),
            timeout_seconds=args.recon_timeout_seconds,
        )
        recon = read_json_result(recon_path, status)
        if recon is None and recon_path.is_file():
            try:
                recon = json.loads(recon_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                recon = None
        if recon is not None:
            results["recon"] = recon
            if recon.get("strict_complete") is not True and status.get("status") == "completed":
                status.update(
                    {
                        "status": "completed_with_errors",
                        "reason": "required reconstruction metrics lack full item coverage",
                        "incomplete_metrics": sorted(
                            recon.get("incomplete_metrics", {}).keys()
                        ),
                    }
                )
        results["_status"]["recon"] = status

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

    report_statuses: dict[str, dict[str, Any]] = {}
    report_results: dict[str, Any] = {}
    if not args.report_evals:
        results["_status"]["report_evals"] = {
            "status": "skipped",
            "reason": "disabled; pass --report_evals to run temporal emotion and representation plots",
        }
    elif not args.subesco_dir:
        reason = "--subesco_dir was not supplied; no implicit dataset path is used"
        for task in ("emotion_temporal_attn", "emotion_temporal_transformer", "repr_tsne_umap_mos"):
            report_statuses[task] = {"status": "skipped", "reason": reason}
        results["report_evals"] = {"_status": report_statuses}
        results["_status"]["report_evals"] = {"status": "skipped", "reason": reason}
    else:
        subesco_dir = pathlib.Path(args.subesco_dir)
        if not subesco_dir.is_dir():
            reason = f"SUBESCO directory does not exist: {subesco_dir}"
            for task in ("emotion_temporal_attn", "emotion_temporal_transformer", "repr_tsne_umap_mos"):
                report_statuses[task] = {"status": "skipped", "reason": reason}
        else:
            attn_path = output_dir / "emotion_temporal_attn.json"
            attn_status = run_command(
                label="SUBESCO attentive-statistics temporal emotion probe",
                command=[
                    args.python_bin, "-m", "eval.eval_emotion_temporal",
                    "--models", "ours", "--ckpt", args.ckpt,
                    "--subesco-dir", str(subesco_dir),
                    "--max-utts", str(args.temporal_max_utts),
                    "--epochs", str(args.temporal_attn_epochs),
                    "--seed", str(args.report_seed), "--out", str(attn_path),
                ],
                step=int(step),
                timeout_seconds=args.probe_timeout_seconds,
            )
            report_statuses["emotion_temporal_attn"] = attn_status
            attn_result = read_json_result(attn_path, attn_status)
            if attn_result is not None:
                report_results["emotion_temporal_attn"] = attn_result

            transformer_path = output_dir / "emotion_temporal_transformer.json"
            transformer_status = run_command(
                label="SUBESCO Transformer temporal emotion probe",
                command=[
                    args.python_bin, "-m", "eval.eval_emotion_transformer",
                    "--models", "ours", "--ckpt", args.ckpt,
                    "--subesco-dir", str(subesco_dir),
                    "--max-utts", str(args.temporal_max_utts),
                    "--epochs", str(args.temporal_transformer_epochs),
                    "--seed", str(args.report_seed), "--out", str(transformer_path),
                ],
                step=int(step),
                timeout_seconds=args.probe_timeout_seconds,
            )
            report_statuses["emotion_temporal_transformer"] = transformer_status
            transformer_result = read_json_result(transformer_path, transformer_status)
            if transformer_result is not None:
                report_results["emotion_temporal_transformer"] = transformer_result

            plot_path = output_dir / "repr_tsne_umap_mos.png"
            plot_metadata_path = output_dir / "repr_tsne_umap_mos.json"
            plot_status = run_command(
                label="MOS-colored representation t-SNE/UMAP",
                command=[
                    args.python_bin, "-m", "eval.eval_repr_cluster",
                    "--source", "subesco", "--models", "ours",
                    "--ckpt", args.ckpt, "--subesco-dir", str(subesco_dir),
                    "--max-utts", str(args.repr_viz_max_utts),
                    "--seed", str(args.report_seed), "--out", str(plot_path),
                    "--metadata-out", str(plot_metadata_path),
                ],
                step=int(step),
                timeout_seconds=args.probe_timeout_seconds,
            )
            report_statuses["repr_tsne_umap_mos"] = plot_status
            plot_result = read_json_result(plot_metadata_path, plot_status)
            if plot_result is not None:
                report_results["repr_tsne_umap_mos"] = plot_result

        report_results["_status"] = report_statuses
        results["report_evals"] = report_results
        report_aggregate = aggregate_status(list(report_statuses.values()))
        results["_status"]["report_evals"] = {"status": report_aggregate}
        if report_aggregate == "skipped":
            results["_status"]["report_evals"]["reason"] = next(
                (
                    status.get("reason", "all report tasks were skipped")
                    for status in report_statuses.values()
                    if status.get("reason")
                ),
                "all report tasks were skipped",
            )

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
