from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from utils.schema import Config


class JsonlLogger:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Dict[str, Any]) -> None:
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def maybe_init_wandb(cfg: "Config", run_id: str, run_dir: str, resume: bool = False):
    wb = cfg.run.wandb
    if not wb.enabled:
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    name = wb.name or run_id
    return wandb.init(
        project=wb.project,
        name=name,
        id=run_id,
        resume="allow" if resume else None,
        dir=run_dir,
        config=cfg.model_dump(),
    )

