from __future__ import annotations

from pathlib import Path
from typing import Iterator

from clae_data.adapters.base import DatasetAdapter
from clae_data.adapters._hf_parquet import hf_snapshot_download, iter_parquet_records
from clae_data.schema import Record


class IndicVoicesAdapter(DatasetAdapter):
    name = "indicvoices"
    language = "bn"
    requires_credentials = ("HF_TOKEN",)

    _REPO_ID = "ai4bharat/indicvoices_r"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "indicvoices"
        hf_snapshot_download(self._REPO_ID, out_dir, allow_patterns="Bengali/*")
        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        # Restrict the parquet glob to the Bengali language subdir so we
        # don't accidentally pick up unrelated splits if the cache grows.
        yield from iter_parquet_records(
            raw_dir=raw_dir,
            dataset_name=self.name,
            language=self.language,
            glob_patterns=("Bengali/**/*.parquet", "**/*.parquet"),
        )
