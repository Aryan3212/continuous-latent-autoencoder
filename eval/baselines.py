from __future__ import annotations

from typing import Any, Dict


def run_encodec_baseline(*, manifest: str) -> Dict[str, Any]:
    try:
        import encodec  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"skipped": True, "reason": f"encodec not available: {exc}"}
    return {"skipped": True, "reason": "encodec baseline not wired in this environment"}


def run_hubert_baseline(*, manifest: str) -> Dict[str, Any]:
    try:
        import fairseq  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"skipped": True, "reason": f"fairseq/hubert not available: {exc}"}
    return {"skipped": True, "reason": "hubert baseline not wired in this environment"}
