from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional


class JsonlLogger:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Dict[str, Any]) -> None:
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def maybe_init_wandb(cfg: Dict[str, Any], run_id: str, run_dir: str, resume: bool = False):
    wb_cfg = (cfg.get("run") or {}).get("wandb") or {}
    if not wb_cfg.get("enabled", False):
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    name = wb_cfg.get("name") or run_id
    return wandb.init(
        project=wb_cfg.get("project", "continuous-latent-ae"),
        name=name,
        id=run_id,
        resume="allow" if resume else None,
        dir=run_dir,
        config=cfg,
    )

