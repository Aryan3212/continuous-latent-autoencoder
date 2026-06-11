from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, List

import yaml

from utils.schema import Config


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(pathlib.Path(str(path)).read_text(encoding="utf-8"))
    base = cfg.get("_base_")
    if base is None:
        return cfg
    base_cfg = _load_raw(base)
    cfg2 = dict(cfg)
    cfg2.pop("_base_", None)
    return _deep_update(base_cfg, cfg2)


def load_config(path: str) -> Config:
    """Load a YAML config, validate against the pydantic schema, and return a
    Config object. Validation runs once at startup so schema errors surface
    with clear messages rather than mid-training KeyError/TypeError.
    """
    raw = _load_raw(path)
    return Config.model_validate(raw)


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply dot-separated KEY=VALUE overrides to a Config and re-validate.

    Overrides format: ``key1.key2=value`` where value is parsed as YAML scalar.
    """
    out = cfg.model_dump()
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be KEY=VALUE, got: {ov}")
        key, val = ov.split("=", 1)
        val_parsed = yaml.safe_load(val)
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val_parsed
    return Config.model_validate(out)
