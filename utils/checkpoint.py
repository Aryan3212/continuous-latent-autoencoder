from __future__ import annotations

import pathlib
import subprocess
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import yaml

if TYPE_CHECKING:
    from utils.schema import Config


def save_checkpoint(
    path: str,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: "Config",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg.model_dump(),
        "extra": extra or {},
    }
    tmp = p.with_suffix(".tmp")
    torch.save(payload, str(tmp))
    tmp.rename(p)


def try_git_hash(cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def save_run_metadata(run_dir: str, cfg: "Config", *, extra: Optional[Dict[str, Any]] = None) -> None:
    p = pathlib.Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
    if extra:
        (p / "run_meta.yaml").write_text(yaml.safe_dump(extra, sort_keys=False), encoding="utf-8")
