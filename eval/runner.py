from __future__ import annotations

import json
import pathlib
import subprocess
import time
from typing import Any, Dict


ERROR_STATUSES = {"failed", "timed_out"}


def aggregate_status(statuses: list[Dict[str, Any]]) -> str:
    """Collapse child statuses without treating an all-skip run as completed."""
    if any(status.get("status") in ERROR_STATUSES for status in statuses):
        return "completed_with_errors"
    if statuses and all(status.get("status") == "skipped" for status in statuses):
        return "skipped"
    return "completed"


def run_command(
    *,
    label: str,
    command: list[str],
    step: int,
    timeout_seconds: int | None = None,
) -> Dict[str, Any]:
    """Run one eval command with visible logs and a JSON-friendly status."""
    started = time.perf_counter()
    status: Dict[str, Any]
    print(f"[Eval step {step}] Starting {label}...", flush=True)
    try:
        subprocess.run(command, check=True, timeout=timeout_seconds)
    except subprocess.CalledProcessError as exc:
        status = {"status": "failed", "returncode": exc.returncode}
    except subprocess.TimeoutExpired:
        status = {"status": "timed_out"}
    except OSError as exc:
        status = {"status": "failed", "reason": str(exc)}
    else:
        status = {"status": "completed"}

    duration = time.perf_counter() - started
    status["duration_seconds"] = duration
    print(
        f"[Eval step {step}] {label} {status['status']} in {duration:.1f}s.",
        flush=True,
    )
    return status


def read_json_result(
    path: pathlib.Path,
    status: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Read a completed command's JSON output or convert it to a failure."""
    if status["status"] != "completed":
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        status.update({"status": "failed", "reason": str(exc)})
        return None
