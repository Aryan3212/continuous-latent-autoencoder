from __future__ import annotations

from pathlib import Path
from typing import Iterator

from clae_data.adapters.base import DatasetAdapter
from clae_data.adapters._hf_parquet import hf_snapshot_download, iter_parquet_records
from clae_data.schema import Record


class KathbathAdapter(DatasetAdapter):
    """Probe-only dataset (eval splits). Records are still emitted; the build
    step is responsible for routing them to the probe manifest rather than
    the pretraining manifest.
    """

    name = "kathbath"
    language = "bn"
    requires_credentials = ("HF_TOKEN",)

    _REPO_ID = "ai4bharat/Kathbath"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "kathbath"
        hf_snapshot_download(
            self._REPO_ID,
            out_dir,
            allow_patterns=["bn/test/*", "bn/valid/*"],
        )
        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        yield from iter_parquet_records(
            raw_dir=raw_dir,
            dataset_name=self.name,
            language=self.language,
            glob_patterns=(
                "bn/test/**/*.parquet",
                "bn/valid/**/*.parquet",
                "**/*.parquet",
            ),
        )
