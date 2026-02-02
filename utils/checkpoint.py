from __future__ import annotations

import hashlib
import pathlib
import subprocess
from typing import Any, Dict, Optional

import torch
import yaml


def save_checkpoint(
    path: str,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg,
        "extra": extra or {},
    }
    torch.save(payload, str(p))


def try_git_hash(cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_run_metadata(run_dir: str, cfg: Dict[str, Any], *, extra: Optional[Dict[str, Any]] = None) -> None:
    p = pathlib.Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if extra:
        (p / "run_meta.yaml").write_text(yaml.safe_dump(extra, sort_keys=False))
