from __future__ import annotations

from typing import Any, Mapping, Optional, TypedDict


class Record(TypedDict, total=False):
    """One audio clip in the unified manifest schema.

    Required at adapter-emit time: ``audio_filepath``, ``dataset``.
    ``audio_filepath`` is absolute on the prep instance and gets rewritten to
    a repo-relative path during the pack step.
    """

    audio_filepath: str
    text: Optional[str]
    duration: Optional[float]
    sample_rate: Optional[int]
    dataset: str
    id: Optional[str]
    speaker_id: Optional[str]
    language: Optional[str]


_REQUIRED: tuple[str, ...] = ("audio_filepath", "dataset")


def validate_record(r: Mapping[str, Any]) -> None:
    """Raise ValueError if ``r`` is missing a required field or has a bad type."""
    for k in _REQUIRED:
        if k not in r or r[k] in (None, ""):
            raise ValueError(f"Record missing required field {k!r}: {dict(r)!r}")
    if not isinstance(r["audio_filepath"], str):
        raise ValueError(
            f"Record.audio_filepath must be str, got {type(r['audio_filepath']).__name__}"
        )
    if not isinstance(r["dataset"], str):
        raise ValueError(
            f"Record.dataset must be str, got {type(r['dataset']).__name__}"
        )
