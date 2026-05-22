from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterator

from clae_data.adapters.base import DatasetAdapter
from clae_data.schema import Record


def _authenticate_kaggle() -> "object":
    # Lazy import: keep clae_data importable without the kaggle package.
    from kaggle.api.kaggle_api_extended import KaggleApi
    from clae_data._creds import KAGGLE_USERNAME, KAGGLE_KEY

    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    api = KaggleApi()
    api.authenticate()
    return api


class BengaliAISpeechAdapter(DatasetAdapter):
    name = "bengaliai_speech"
    language = "bn"
    requires_credentials = ("KAGGLE_USERNAME", "KAGGLE_KEY")

    _SLUG = "bengaliai-speech"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "bengaliai_speech"
        out_dir.mkdir(parents=True, exist_ok=True)

        # If we see the unpacked structure, treat as done.
        if (out_dir / "train.csv").exists() and (out_dir / "train_mp3s").exists():
            return out_dir

        api = _authenticate_kaggle()
        print(f"[bengaliai_speech] downloading kaggle competition {self._SLUG}")
        api.competition_download_files(self._SLUG, path=str(out_dir), quiet=False)

        # Kaggle usually drops <slug>.zip; fall back to the first zip found.
        zip_path = out_dir / f"{self._SLUG}.zip"
        if not zip_path.exists():
            zips = sorted(out_dir.glob("*.zip"))
            if not zips:
                raise FileNotFoundError(
                    f"[bengaliai_speech] no zip found in {out_dir!s} after download"
                )
            zip_path = zips[0]

        print(f"[bengaliai_speech] extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        zip_path.unlink(missing_ok=True)
        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        # pandas is heavyweight; lazy-import so import-time stays cheap.
        import pandas as pd

        csv_path = raw_dir / "train.csv"
        audio_root = raw_dir / "train_mp3s"
        if not csv_path.exists():
            print(f"[bengaliai_speech] missing {csv_path!s}")
            return

        df = pd.read_csv(csv_path)
        # CSV schema: id, sentence, split.
        for _, row in df.iterrows():
            audio_id = str(row["id"])
            audio_path = audio_root / f"{audio_id}.mp3"
            if not audio_path.exists():
                continue
            text = row.get("sentence", "") if "sentence" in row.index else ""
            rec: Record = {
                "audio_filepath": str(audio_path),
                "text": str(text) if text is not None else None,
                "duration": None,
                "sample_rate": None,
                "dataset": self.name,
                "id": audio_id,
                "speaker_id": None,
                "language": self.language,
            }
            yield rec
