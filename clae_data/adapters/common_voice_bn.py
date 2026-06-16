from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Iterator

from clae_data.adapters.base import DatasetAdapter
from clae_data.schema import Record


class CommonVoiceBnAdapter(DatasetAdapter):
    """Mozilla Common Voice (Scripted Speech) — Bengali.

    Distributed via the Mozilla Data Collective platform (Common Voice left HF
    in Oct 2025). We pull it with the official ``datacollective`` SDK, which is
    CC0-licensed and needs no per-competition rules acceptance — just an
    ``MDC_API_KEY``. On disk it's the standard Common Voice layout:
    ``clips/*.mp3`` plus ``*.tsv`` manifests (we read ``validated.tsv``).
    """

    name = "common_voice_bn"
    language = "bn"
    requires_credentials = ("MDC_API_KEY",)

    # Tail of the dataset URL: mozilladatacollective.com/datasets/<id>
    _DATASET_ID = "cmn3ipo8b00ejmi079e8upl2k"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "common_voice_bn"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Already extracted? (idempotent re-run) — bail before hitting the API,
        # which is rate-limited to 30 presigned-URL requests/day per org.
        if self._find_validated_tsv(out_dir) is not None:
            print(f"[common_voice_bn] already extracted under {out_dir!s}")
            return out_dir

        try:
            from clae_data._creds import MDC_API_KEY
        except ImportError as e:
            raise SystemExit(
                "[common_voice_bn] MDC_API_KEY missing from clae_data/_creds.py; "
                "add it from your Mozilla Data Collective Account -> Credentials."
            ) from e

        os.environ["MDC_API_KEY"] = MDC_API_KEY
        # Contain the SDK's download under our data root (default is ~/.mozdata).
        os.environ["MDC_DOWNLOAD_PATH"] = str(out_dir)

        # Import after env is set so the SDK picks up our config.
        from datacollective import download_dataset

        print(f"[common_voice_bn] download_dataset {self._DATASET_ID} -> {out_dir}")
        download_dataset(self._DATASET_ID)

        # The SDK may or may not auto-extract the tar.gz; do it ourselves if the
        # tsv isn't visible yet but an archive is present.
        if self._find_validated_tsv(out_dir) is None:
            for tar_path in sorted(out_dir.rglob("*.tar.gz")):
                print(f"[common_voice_bn] extracting {tar_path.name}")
                with tarfile.open(tar_path, "r:gz") as tf:
                    tf.extractall(out_dir)
                # Keep the archive; deleting risks re-download against the daily cap.

        return out_dir

    @staticmethod
    def _find_validated_tsv(raw_dir: Path) -> Path | None:
        # Common Voice nests under cv-corpus-<ver>-<date>/<locale>/; glob for it.
        for name in ("validated.tsv", "train.tsv"):
            hits = sorted(raw_dir.rglob(name))
            if hits:
                return hits[0]
        return None

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        import pandas as pd

        tsv = self._find_validated_tsv(raw_dir)
        if tsv is None:
            print(f"[common_voice_bn] no validated.tsv/train.tsv under {raw_dir!s}")
            return
        cv_dir = tsv.parent
        clips_dir = cv_dir / "clips"

        df = pd.read_csv(tsv, sep="\t", low_memory=False)
        # Common Voice tsv: client_id, path, sentence, up_votes, down_votes, ...
        if "path" not in df.columns or "sentence" not in df.columns:
            print(f"[common_voice_bn] unexpected tsv columns: {list(df.columns)}")
            return

        for _, row in df.iterrows():
            rel = str(row["path"])
            audio_path = clips_dir / rel
            if not audio_path.exists():
                # Some versions already include the clips/ prefix in `path`.
                alt = cv_dir / rel
                if alt.exists():
                    audio_path = alt
                else:
                    continue
            sentence = row.get("sentence")
            text = None if sentence is None or pd.isna(sentence) else str(sentence)
            client = row.get("client_id")
            speaker = None if client is None or pd.isna(client) else str(client)
            rec: Record = {
                "audio_filepath": str(audio_path),
                "text": text,
                "duration": None,
                "sample_rate": None,
                "dataset": self.name,
                "id": Path(rel).stem,
                "speaker_id": speaker,
                "language": self.language,
            }
            yield rec
