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
