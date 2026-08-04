#!/usr/bin/env python3
"""Create matched representation-diagnostic plots from ablation JSONL logs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training_diagnostics import COLLAPSED_DIM_STD_THRESHOLD


CONDITION_ARGS = {
    "full": "Full R + J + VISReg",
    "reconstruction_only": "Reconstruction only R",
    "representation_only": "Representation only J + VISReg",
    "no_mhc": "No mHC",
    "no_decoder_corruption": "No decoder corruption",
}
MILESTONES = (10_000, 25_000, 50_000)

SPACE_METRICS = {
    "z": {
        "rank": ["latent/z/effective_rank"],
        "std": [
            "latent/z/dim_std_min",
            "latent/z/dim_std_p05",
            "latent/z/dim_std_median",
            "latent/z/dim_std_p95",
            "latent/z/dim_std_max",
        ],
        "covariance": [
            "latent/z/cov_offdiag_abs_mean",
            "latent/z/cov_offdiag_rms",
            "latent/z/isotropy_ratio",
        ],
        "collapse": ["latent/z/collapsed_dim_frac"],
        "similarity": [
            "sim/pos_frame_mse",
            "sim/neg_frame_mse",
            "sim/pos_utt_mse",
            "sim/neg_utt_mse",
        ],
    },
    "p": {
        "rank": ["projector/p/effective_rank"],
        "std": [
            "projector/p/dim_std_min",
            "projector/p/dim_std_p05",
            "projector/p/dim_std_median",
            "projector/p/dim_std_p95",
            "projector/p/dim_std_max",
        ],
        "covariance": [
            "projector/p/cov_offdiag_abs_mean",
            "projector/p/cov_offdiag_rms",
            "projector/p/isotropy_ratio",
        ],
        "collapse": ["projector/p/collapsed_dim_frac"],
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument, label in CONDITION_ARGS.items():
        parser.add_argument(
            f"--{argument.replace('_', '-')}",
            required=True,
            type=Path,
            help=f"Run directory or train.jsonl for {label}",
        )
    parser.add_argument(
        "--supplementary-25hz",
        type=Path,
        default=None,
        help="Optional 25 Hz run, rendered separately rather than mixed into matched plots.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=None,
        help="Exact common steps to plot (default: all common metric steps).",
    )
    return parser.parse_args()


def _resolve_log(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return path, path.parent.parent if path.parent.name == "logs" else path.parent
    candidate = path / "logs" / "train.jsonl"
    if candidate.is_file():
        return candidate, path
    candidate = path / "train.jsonl"
    if candidate.is_file():
        return candidate, path
    raise FileNotFoundError(f"no train.jsonl found under {path}")


def load_merged_rows(path: Path) -> tuple[dict[int, dict[str, Any]], Path, str]:
    log_path, run_dir = _resolve_log(path)
    rows: dict[int, dict[str, Any]] = {}
    with log_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {log_path}:{line_number}: {error}") from error
            if "step" not in row:
                raise ValueError(f"missing step at {log_path}:{line_number}")
            step = int(row["step"])
            rows.setdefault(step, {}).update(row)
    if not rows:
        raise ValueError(f"no rows found in {log_path}")

    config_path = run_dir / "config.yaml"
    config_name = "unknown"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        config_name = str(
            config.get("run", {}).get("wandb", {}).get("name")
            or config_path.name
        )
    return rows, run_dir, config_name


def _required_keys(space: str) -> set[str]:
    return {key for keys in SPACE_METRICS[space].values() for key in keys}


def common_complete_steps(
    runs: dict[str, dict[int, dict[str, Any]]],
    *,
    space: str,
    requested_steps: list[int] | None,
) -> list[int]:
    required = _required_keys(space)
    complete_by_run = {
        condition: {
            step for step, row in rows.items() if required.issubset(row)
        }
        for condition, rows in runs.items()
    }
    common = set.intersection(*complete_by_run.values())
    if requested_steps is not None:
        missing = sorted(set(requested_steps) - common)
        if missing:
            details = {
                condition: sorted(set(missing) - steps)
                for condition, steps in complete_by_run.items()
                if set(missing) - steps
            }
            raise ValueError(
                f"requested {space}-space steps lack required metrics: {details}"
            )
        selected = sorted(set(requested_steps))
    else:
        selected = sorted(common)
    if not selected:
        raise ValueError(f"no common complete {space}-space diagnostic steps")
    missing_milestones = sorted(set(MILESTONES) - set(selected))
    if missing_milestones:
        raise ValueError(
            f"{space}-space plots require milestone steps {MILESTONES}; missing {missing_milestones}"
        )
    return selected


def _mark_milestones(axes: list[plt.Axes]) -> None:
    for axis in axes:
        for step in MILESTONES:
            axis.axvline(step, color="0.8", linewidth=0.8, linestyle=":", zorder=0)
        axis.grid(alpha=0.2)
        axis.set_xlabel("optimizer step")


def plot_space(
    *,
    space: str,
    runs: dict[str, dict[int, dict[str, Any]]],
    steps: list[int],
    output_path: Path,
    title_suffix: str = "",
    condition_labels: dict[str, str] | None = None,
) -> set[str]:
    panels = SPACE_METRICS[space]
    labels = condition_labels or CONDITION_ARGS
    figure, axes_array = plt.subplots(
        len(panels), 1, figsize=(10, 3.1 * len(panels)), sharex=True, constrained_layout=True
    )
    axes = list(axes_array) if hasattr(axes_array, "__len__") else [axes_array]
    used_keys: set[str] = set()

    for axis, (panel, keys) in zip(axes, panels.items()):
        for condition, label in labels.items():
            rows = runs[condition]
            if panel == "std":
                median = [rows[step][keys[2]] for step in steps]
                p05 = [rows[step][keys[1]] for step in steps]
                p95 = [rows[step][keys[3]] for step in steps]
                minimum = [rows[step][keys[0]] for step in steps]
                maximum = [rows[step][keys[4]] for step in steps]
                line = axis.plot(steps, median, label=label)[0]
                axis.fill_between(steps, p05, p95, color=line.get_color(), alpha=0.12)
                axis.plot(steps, minimum, color=line.get_color(), alpha=0.35, linewidth=0.7)
                axis.plot(steps, maximum, color=line.get_color(), alpha=0.35, linewidth=0.7)
            else:
                for key in keys:
                    short_name = key.rsplit("/", 1)[-1]
                    axis.plot(
                        steps,
                        [rows[step][key] for step in steps],
                        label=f"{label} · {short_name}" if len(keys) > 1 else label,
                    )
            used_keys.update(keys)
        axis.set_ylabel(panel.replace("_", " "))
        axis.legend(fontsize=7, ncol=2)

    _mark_milestones(axes)
    space_name = "encoder latent z" if space == "z" else "VISReg projector p"
    figure.suptitle(f"Ablation representation diagnostics — {space_name}{title_suffix}")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return used_keys


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_paths = {
        condition: getattr(args, condition)
        for condition in CONDITION_ARGS
    }
    loaded = {condition: load_merged_rows(path) for condition, path in run_paths.items()}
    runs = {condition: value[0] for condition, value in loaded.items()}
    complete_steps = {
        space: common_complete_steps(runs, space=space, requested_steps=args.steps)
        for space in ("z", "p")
    }
    common_steps = sorted(set(complete_steps["z"]) & set(complete_steps["p"]))
    selected_steps = {space: common_steps for space in ("z", "p")}
    used_keys: set[str] = set()
    for space in ("z", "p"):
        used_keys.update(
            plot_space(
                space=space,
                runs=runs,
                steps=selected_steps[space],
                output_path=args.output_dir / f"ablation_{space}_diagnostics.png",
            )
        )

    supplementary: dict[str, Any] | None = None
    if args.supplementary_25hz is not None:
        rows, run_dir, config_name = load_merged_rows(args.supplementary_25hz)
        single_run = {"full": rows}
        supplementary_complete_steps = {
            space: common_complete_steps(
                single_run, space=space, requested_steps=args.steps
            )
            for space in ("z", "p")
        }
        supplementary_common_steps = sorted(
            set(supplementary_complete_steps["z"])
            & set(supplementary_complete_steps["p"])
        )
        supplementary_steps = {
            space: supplementary_common_steps for space in ("z", "p")
        }
        for space in ("z", "p"):
            used_keys.update(
                plot_space(
                    space=space,
                    runs=single_run,
                    steps=supplementary_steps[space],
                    output_path=args.output_dir / f"supplementary_25hz_{space}_diagnostics.png",
                    title_suffix=" (25 Hz supplementary)",
                    condition_labels={"full": "25 Hz full objective"},
                )
            )
        supplementary = {
            "run_directory": str(run_dir),
            "config_name": config_name,
            "selected_steps": supplementary_steps,
        }

    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_runs": {
            condition: {
                "run_directory": str(value[1]),
                "config_name": value[2],
            }
            for condition, value in loaded.items()
        },
        "metric_keys": sorted(used_keys),
        "selected_steps": selected_steps,
        "collapsed_dim_std_threshold": COLLAPSED_DIM_STD_THRESHOLD,
        "supplementary_25hz": supplementary,
    }
    with (args.output_dir / "diagnostic_plot_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
