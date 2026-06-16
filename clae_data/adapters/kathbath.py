from __future__ import annotations

from pathlib import Path
from typing import Iterator

from clae_data.adapters.base import DatasetAdapter
from clae_data.adapters._hf_parquet import hf_snapshot_download, iter_parquet_records
from clae_data.schema import Record


class KathbathAdapter(DatasetAdapter):
    """Bengali eval/probe corpus (valid-* shards). pack.py does NOT
    special-case kathbath — there is no automatic routing to a probe
    manifest. To keep it as a held-out eval set, it must be excluded from
    the pretraining ``--datasets`` list; that's why it's left out of the
    Makefile default. It remains registered here so it can still be
    downloaded explicitly via ``DATASETS=kathbath``.
    """

    name = "kathbath"
    language = "bn"
    requires_credentials = ("HF_TOKEN",)

    _REPO_ID = "ai4bharat/Kathbath"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "kathbath"
        # Bengali subset is under `bengali/` as flat shards; no test split
        # exists, so use valid-* as the held-out probe/eval set.
        hf_snapshot_download(
            self._REPO_ID,
            out_dir,
            allow_patterns=["bengali/valid-*.parquet"],
        )
        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        yield from iter_parquet_records(
            raw_dir=raw_dir,
            dataset_name=self.name,
            language=self.language,
            glob_patterns=(
                "bengali/valid-*.parquet",
                "bengali/**/valid-*.parquet",
            ),
        )
