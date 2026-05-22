from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterator

from clae_data.schema import Record


class DatasetAdapter(abc.ABC):
    """Per-dataset surface: download raw archives, yield unified Records."""

    name: str
    language: str = "bn"
    # Names of credential variables expected to be present (sourced from
    # clae_data._creds). Adapters use this to fail fast before a download.
    requires_credentials: tuple[str, ...] = ()

    @abc.abstractmethod
    def download(self, dest_root: Path) -> Path:
        """Download raw archives under ``dest_root``.

        Must be idempotent: a second call with the same ``dest_root`` should
        be a no-op (or only re-fetch missing parts). Returns the raw-data
        directory used as input to ``iter_records``.
        """

    @abc.abstractmethod
    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        """Yield one ``Record`` per audio clip.

        ``audio_filepath`` should be an absolute path on the prep instance.
        The pack step is responsible for transcoding and rewriting paths.
        """
