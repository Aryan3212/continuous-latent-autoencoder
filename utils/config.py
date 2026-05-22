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
    cfg = yaml.safe_load(pathlib.Path(str(path)).read_text())
    base = cfg.get("_base_")
    if base is None:
        return cfg
    base_cfg = _load_raw(base)
    cfg2 = dict(cfg)
    cfg2.pop("_base_", None)
    return _deep_update(base_cfg, cfg2)


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config, validate it against the pydantic schema, and
    return a type-coerced dict so existing call sites continue to work.

    Validation runs once at startup — schema errors are raised here with
    clear messages, instead of `KeyError`/`TypeError` mid-training.
    """
    raw = _load_raw(path)
    model = Config.model_validate(raw)
    # Return a plain dict (recursive) so train.py's existing cfg["..."] access
    # keeps working. The dict has been coerced to the schema types.
    return model.model_dump()


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Overrides format: key1.key2=value, where value is parsed as YAML scalar.
    Re-validates the resulting dict against the schema so type errors in
    overrides also fail fast.
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
    # Re-validate after applying overrides so bad CLI args fail fast.
    return Config.model_validate(out).model_dump()
