from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, List, Tuple

import yaml


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    path = str(path)
    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    base = cfg.get("_base_")
    if base is None:
        return cfg
    base_cfg = load_config(base)
    cfg2 = dict(cfg)
    cfg2.pop("_base_", None)
    return _deep_update(base_cfg, cfg2)


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Overrides format: key1.key2=value, where value is parsed as YAML scalar.
    """
    out = copy.deepcopy(cfg)
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
    return out

