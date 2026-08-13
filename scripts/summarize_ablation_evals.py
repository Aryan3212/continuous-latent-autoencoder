#!/usr/bin/env python3
"""Consolidate fixed 25k six-condition evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ERROR_STATUSES = {"failed", "timed_out", "completed_with_errors"}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _format(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _status(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "missing"
    statuses = summary.get("_status", {})
    if any(item.get("status") in ERROR_STATUSES for item in statuses.values()):
        return "completed_with_errors"
    if statuses and all(item.get("status") == "skipped" for item in statuses.values()):
        return "skipped"
    return "completed"


def _probe_metric(summary: dict[str, Any], probe: str, *keys: str) -> float | None:
    value: Any = summary.get("probes", {}).get(probe, {})
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _number(value)


def _recon_metric(summary: dict[str, Any], key: str) -> float | None:
    recon = summary.get("recon", {})
    if recon.get("strict_complete") is not True:
        return None
    return _number(recon.get("aggregate", {}).get(key, recon.get(key)))


def _report_metric(summary: dict[str, Any], task: str, key: str) -> float | None:
    return _number(
        summary.get("report_evals", {}).get(task, {}).get("results", {}).get("ours", {}).get(key)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate per-condition eval.run_all summaries."
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--mimi_result", type=Path, default=None)
    parser.add_argument("--listening_dir", type=Path, default=None)
    parser.add_argument(
        "--condition",
        action="append",
        nargs=5,
        metavar=("LABEL", "CONFIG", "RUN_ID", "CHECKPOINT", "EVAL_OUT_DIR"),
        required=True,
        help="One condition to include; repeat for every condition.",
    )
    args = parser.parse_args()
    if args.step < 0:
        parser.error("--step must be non-negative")

    conditions: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for label, config, run_id, checkpoint, evaluation_dir in args.condition:
        summary_path = Path(evaluation_dir) / f"step_{args.step}" / "summary.json"
        read_error: str | None = None
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            summary = None
        except OSError as exc:
            summary = None
            read_error = f"could not read summary: {exc}"
        except json.JSONDecodeError as exc:
            summary = None
            read_error = f"invalid JSON: {exc}"

        condition: dict[str, Any] = {
            "label": label,
            "config": config,
            "run_id": run_id,
            "checkpoint": checkpoint,
            "evaluation_summary": str(summary_path),
            "status": _status(summary),
        }
        if read_error:
            condition["read_error"] = read_error
        if summary is not None:
            condition["results"] = summary
        conditions.append(condition)

        summary_data = summary or {}
        table_rows.append(
            {
                "label": label,
                "status": condition["status"],
                "stft": _recon_metric(summary_data, "stft"),
                "wav_l1": _recon_metric(summary_data, "wav_l1"),
                "si_sdr_db": _recon_metric(summary_data, "si_sdr_db"),
                "stoi": _recon_metric(summary_data, "stoi"),
                "estoi": _recon_metric(summary_data, "estoi"),
                "pesq_wb": _recon_metric(summary_data, "pesq_wb"),
                "temporal_attn_f1": _report_metric(
                    summary_data, "emotion_temporal_attn", "macro_f1"
                ),
                "temporal_transformer_f1": _report_metric(
                    summary_data, "emotion_temporal_transformer", "macro_f1"
                ),
                "asr_cer": _probe_metric(summary_data, "asr", "dev", "cer"),
            }
        )

    mimi: dict[str, Any] | None = None
    if args.mimi_result is not None:
        try:
            mimi = json.loads(args.mimi_result.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            mimi = None
    listening_ready = bool(
        args.listening_dir
        and (args.listening_dir / "public" / "trials.csv").is_file()
        and (args.listening_dir / "private" / "condition_key.json").is_file()
    )
    payload = {
        "step": args.step,
        "conditions": conditions,
        "baselines": {"mimi_8q_1.1kbps": mimi},
        "listening_study": {
            "status": "prepared" if listening_ready else "missing_or_skipped",
            "directory": str(args.listening_dir) if args.listening_dir else None,
            "human_ratings_collected": False,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 25k packed ablation evaluation results (six conditions)",
        "",
        "`—` means that an evaluation was skipped, failed, or did not emit that metric.",
        "",
        "| condition | status | STFT ↓ | L1 ↓ | SI-SDR dB ↑ | STOI ↑ | ESTOI ↑ | PESQ-WB ↑ | attn emotion F1 ↑ | Transformer emotion F1 ↑ | ASR CER ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            "| {label} | {status} | {stft} | {wav_l1} | {si_sdr_db} | "
            "{stoi} | {estoi} | {pesq_wb} | {temporal_attn_f1} | "
            "{temporal_transformer_f1} | {asr_cer} |".format(
                label=row["label"],
                status=row["status"],
                stft=_format(row["stft"]),
                wav_l1=_format(row["wav_l1"]),
                si_sdr_db=_format(row["si_sdr_db"]),
                stoi=_format(row["stoi"]),
                estoi=_format(row["estoi"]),
                pesq_wb=_format(row["pesq_wb"]),
                temporal_attn_f1=_format(row["temporal_attn_f1"]),
                temporal_transformer_f1=_format(row["temporal_transformer_f1"]),
                asr_cer=_format(row["asr_cer"]),
            )
        )
    if mimi is not None and mimi.get("strict_complete") is True:
        aggregate = mimi.get("aggregate", {})
        lines.append(
            "| Mimi 8q / 1.1 kbps | baseline | {stft} | {wav_l1} | {si_sdr_db} | "
            "{stoi} | {estoi} | {pesq_wb} | — | — | — |".format(
                stft=_format(_number(aggregate.get("stft"))),
                wav_l1=_format(_number(aggregate.get("wav_l1"))),
                si_sdr_db=_format(_number(aggregate.get("si_sdr_db"))),
                stoi=_format(_number(aggregate.get("stoi"))),
                estoi=_format(_number(aggregate.get("estoi"))),
                pesq_wb=_format(_number(aggregate.get("pesq_wb"))),
            )
        )
    elif mimi is not None:
        lines.append("| Mimi 8q / 1.1 kbps | incomplete coverage | — | — | — | — | — | — | — | — | — |")
    lines.extend([
        "",
        "Each condition's complete raw result and status record is embedded in `results.json` "
        "and stored beside its checkpoint evaluation under `step_25000/summary.json`.",
        "The listening-study status describes stimulus preparation only; no human ratings are collected automatically.",
        "",
    ])
    (args.out_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
