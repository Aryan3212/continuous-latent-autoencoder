from __future__ import annotations

from pathlib import Path
from typing import Iterator

from clae_data.adapters.base import DatasetAdapter
from clae_data.adapters._hf_parquet import hf_snapshot_download, iter_parquet_records
from clae_data.schema import Record


class SubakKoAdapter(DatasetAdapter):
    name = "subak_ko"
    language = "bn"
    requires_credentials = ("HF_TOKEN",)

    _REPO_ID = "SUST-CSE-Speech/SUBAK.KO"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "subak_ko"
        # SUBAK.KO is single-language; no allow_patterns filter.
        hf_snapshot_download(self._REPO_ID, out_dir, allow_patterns=None)
        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        yield from iter_parquet_records(
            raw_dir=raw_dir,
            dataset_name=self.name,
            language=self.language,
        )
