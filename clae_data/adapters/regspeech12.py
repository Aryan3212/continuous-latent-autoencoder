from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterator

from clae_data.adapters.base import DatasetAdapter
from clae_data.schema import Record


def _authenticate_kaggle() -> "object":
    from kaggle.api.kaggle_api_extended import KaggleApi
    from clae_data._creds import KAGGLE_USERNAME, KAGGLE_KEY

    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    api = KaggleApi()
    api.authenticate()
    return api


class RegSpeech12Adapter(DatasetAdapter):
    name = "regspeech12"
    language = "bn"
    requires_credentials = ("KAGGLE_USERNAME", "KAGGLE_KEY")

    _SLUG = "mdrezuwanhassan/regspeech12"
    _SPLITS: tuple[str, ...] = ("train", "valid", "test")

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "regspeech12"
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "train.xlsx").exists():
            return out_dir

        api = _authenticate_kaggle()
        print(f"[regspeech12] downloading kaggle dataset {self._SLUG}")
        api.dataset_download_files(
            self._SLUG, path=str(out_dir), quiet=False, unzip=False
        )

        zip_name = f"{self._SLUG.split('/')[1]}.zip"
        zip_path = out_dir / zip_name
        if not zip_path.exists():
            zips = sorted(out_dir.glob("*.zip"))
            if not zips:
                raise FileNotFoundError(
                    f"[regspeech12] no zip found in {out_dir!s} after download"
                )
            zip_path = zips[0]

        print(f"[regspeech12] extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        zip_path.unlink(missing_ok=True)
        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        import pandas as pd

        for split in self._SPLITS:
            xlsx = raw_dir / f"{split}.xlsx"
            audio_root = raw_dir / split
            if not xlsx.exists():
                print(f"[regspeech12] missing {xlsx!s} — skipping split {split}")
                continue
            try:
                df = pd.read_excel(xlsx)
            except Exception as e:
                print(f"[regspeech12] failed to read {xlsx!s}: {e}")
                continue

            # Heuristic column resolution; the original script does the same.
            cols = list(df.columns)
            id_candidates = ["id", "file_name", "filename", "audio", "path", cols[0]]
            text_candidates = [
                "sentence",
                "text",
                "transcript",
                "transcription",
                cols[1] if len(cols) > 1 else cols[0],
            ]
            id_col = next((c for c in id_candidates if c in df.columns), cols[0])
            text_col = next(
                (c for c in text_candidates if c in df.columns),
                cols[1] if len(cols) > 1 else cols[0],
            )

            for _, row in df.iterrows():
                file_id = str(row[id_col])
                text = str(row[text_col]) if row[text_col] is not None else None

                # Filename in the sheet may or may not carry an extension.
                audio_path = audio_root / file_id
                if not audio_path.exists():
                    found = None
                    for ext in (".wav", ".mp3", ".flac"):
                        candidate = audio_root / f"{file_id}{ext}"
                        if candidate.exists():
                            found = candidate
                            break
                    if found is None:
                        continue
                    audio_path = found

                rec: Record = {
                    "audio_filepath": str(audio_path),
                    "text": text,
                    "duration": None,
                    "sample_rate": None,
                    "dataset": self.name,
                    "id": file_id,
                    "speaker_id": None,
                    "language": self.language,
                }
                yield rec
