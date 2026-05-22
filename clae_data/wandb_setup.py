"""Glue: surface ``WANDB_API_KEY`` from ``clae_data._creds`` into the env.

Called once at training-script startup before any ``wandb.init``. Keeps the
wandb library env-driven (its normal interface) without changing call sites
in ``utils/logging.py``.
"""
from __future__ import annotations

import os


_PLACEHOLDER = "<paste-here>"


def setup_wandb_env() -> None:
    """Set ``WANDB_API_KEY`` from ``_creds.WANDB_API_KEY`` if not already set.

    Idempotent. Respects a pre-existing ``WANDB_API_KEY`` env var (user override).
    Prints a warning when the creds file still holds the placeholder value, so
    wandb will fall back to its offline / interactive flow.
    """
    if os.environ.get("WANDB_API_KEY"):
        return

    try:
        from clae_data._creds import WANDB_API_KEY
    except Exception as e:
        print(f"[wandb_setup] could not import _creds.WANDB_API_KEY: {e}")
        return

    if not WANDB_API_KEY or WANDB_API_KEY == _PLACEHOLDER:
        print(
            "[wandb_setup] WANDB_API_KEY not configured in clae_data/_creds.py "
            "(placeholder value). wandb will fall back to offline / interactive."
        )
        return

    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
