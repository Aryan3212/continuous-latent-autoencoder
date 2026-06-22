"""Combined data + artifact housekeeping utility (one file, no package).

Everything for moving datasets and checkpoints to/from HF Hub, plus the
per-source dataset adapters, lives here. Run from the repo root:

    python scripts/housekeeping.py <subcommand> [args...]

Subcommands:
    download             Download raw archives for the given adapters.
    build                Pack records into a staging dir (audio + manifests).
    audit                Probe rows in staging manifests (debug; build runs audit too).
    push                 Upload a staging dir to a HF dataset repo.
    fetch                Snapshot-download a packed HF dataset repo (train-side pull).
    pack-and-push        Convenience: build + push in one shot (prep instance).
    publish-checkpoint   Upload a ``last.pt`` + model card to a HF model repo.

Credentials come straight from the environment (sourced from .env by setup.sh):
``HF_TOKEN`` (all HF ops), ``KAGGLE_USERNAME``/``KAGGLE_KEY`` (regspeech12,
bengaliai_speech), ``MDC_API_KEY`` (common_voice_bn). A missing key is a hard
``KeyError``.

The adapter pattern is preserved: each source is a ``DatasetAdapter`` subclass
with ``download()`` + ``iter_records()``. Heavy deps (torch/torchaudio/
soundfile/pandas/pyarrow/huggingface_hub/...) are imported lazily inside the
functions that need them so the bare CLI stays import-light.
"""
from __future__ import annotations

import abc
import argparse
import datetime as _dt
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence, TypedDict

# Allow running directly as `python scripts/housekeeping.py` — put the repo root
# on sys.path so repo-root imports (schema, config, ...) resolve regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =========================================================================== #
# Schema
# =========================================================================== #


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


# =========================================================================== #
# Adapter base
# =========================================================================== #


class DatasetAdapter(abc.ABC):
    """Per-dataset surface: download raw archives, yield unified Records."""

    name: str
    language: str = "bn"

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


# =========================================================================== #
# Shared helper: HF datasets distributed as parquet with inline audio bytes
# (used by IndicVoices, SUBAK.KO, Shrutilipi, Kathbath)
# =========================================================================== #


def hf_snapshot_download(
    repo_id: str,
    dest_dir: Path,
    allow_patterns: Optional[Sequence[str] | str] = None,
) -> Path:
    """Idempotent snapshot_download into ``dest_dir``.

    huggingface_hub already short-circuits on cache hits, so re-running is
    cheap. We only wrap to consistently pass HF_TOKEN.
    """
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)

    # A `.download.done` marker is written only after snapshot_download returns
    # cleanly, so its presence means the full snapshot landed. An interrupted
    # download leaves no marker and the next run refetches. (snapshot_download
    # already short-circuits per-file on cache hits, so a re-run is cheap even
    # without the marker — the marker is just an unambiguous "fully done" flag.)
    marker = dest_dir / ".download.done"
    if marker.exists():
        print(f"[hf] {repo_id} already fully downloaded -> {dest_dir!s}")
        return dest_dir

    print(f"[hf] snapshot_download {repo_id} -> {dest_dir!s}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest_dir),
        allow_patterns=allow_patterns,
        token=os.environ["HF_TOKEN"],
    )
    marker.touch()
    return dest_dir


def iter_parquet_records(
    raw_dir: Path,
    dataset_name: str,
    language: str,
    extract_subdir: str = "extracted",
    glob_patterns: Sequence[str] = ("**/*.parquet",),
) -> Iterator[Record]:
    """Walk parquet files under ``raw_dir``, extract inline audio, yield Records.

    Recognises the standard HF audio-dataset shape where each row has an
    ``audio`` (or ``speech``) column containing ``{"bytes": ..., "path": ...}``.
    Extracted files land in ``<raw_dir>/<extract_subdir>/<id>.<ext>``.
    """
    import pyarrow.parquet as pq

    extract_root = raw_dir / extract_subdir
    extract_root.mkdir(parents=True, exist_ok=True)

    parquet_files: list[Path] = []
    for pattern in glob_patterns:
        parquet_files.extend(sorted(raw_dir.glob(pattern)))
    # De-dupe while preserving order.
    seen: set[Path] = set()
    parquet_files = [p for p in parquet_files if not (p in seen or seen.add(p))]
    if not parquet_files:
        print(f"[{dataset_name}] no parquet files found under {raw_dir!s}")
        return

    counter = 0
    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
        except Exception as e:
            print(f"[{dataset_name}] failed to read {pf!s}: {e}")
            continue

        cols = table.column_names
        audio_col = next((c for c in ("audio", "speech", "audio_filepath") if c in cols), None)
        if audio_col is None:
            print(f"[{dataset_name}] no audio column in {pf!s}; cols={cols}")
            continue
        text_col = next(
            (c for c in ("text", "sentence", "transcript", "transcription") if c in cols),
            None,
        )
        id_col = next((c for c in ("id", "file", "path", "utt_id") if c in cols), None)
        spk_col = next(
            (c for c in ("speaker_id", "speaker", "client_id") if c in cols), None
        )

        n = table.num_rows
        # Materialise once to avoid per-cell pyarrow scalar overhead.
        audio_data = table[audio_col].to_pylist()
        text_data = table[text_col].to_pylist() if text_col else [None] * n
        id_data = table[id_col].to_pylist() if id_col else [None] * n
        spk_data = table[spk_col].to_pylist() if spk_col else [None] * n

        for i in range(n):
            counter += 1
            audio_field = audio_data[i]
            audio_bytes: Optional[bytes] = None
            inline_path: Optional[str] = None
            if isinstance(audio_field, dict):
                b = audio_field.get("bytes")
                if isinstance(b, (bytes, bytearray)):
                    audio_bytes = bytes(b)
                p = audio_field.get("path")
                if isinstance(p, str) and p:
                    inline_path = p
            elif isinstance(audio_field, (bytes, bytearray)):
                audio_bytes = bytes(audio_field)

            file_id_raw = id_data[i]
            if file_id_raw:
                file_id = Path(str(file_id_raw)).stem
            elif inline_path:
                file_id = Path(inline_path).stem
            else:
                file_id = f"{dataset_name}_{counter:09d}"

            # Pick an extension that matches the inline path when known so
            # downstream torchaudio dispatch isn't confused. Default to .flac
            # (lossless; safe for unknown payload).
            ext = ".flac"
            if inline_path:
                guess = Path(inline_path).suffix.lower()
                if guess in (".flac", ".wav", ".mp3", ".ogg", ".opus"):
                    ext = guess

            audio_path = extract_root / f"{file_id}{ext}"
            if not audio_path.exists():
                if audio_bytes is None:
                    # Some datasets reference a path-on-disk relative to the
                    # parquet's directory rather than inlining bytes.
                    if inline_path:
                        candidate = (pf.parent / inline_path).resolve()
                        if candidate.exists():
                            audio_path = candidate
                        else:
                            continue
                    else:
                        continue
                else:
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)

            text_val = text_data[i]
            text: Optional[str]
            if text_val is None:
                text = None
            else:
                text = str(text_val) if text_val != "" else ""

            spk_val = spk_data[i]
            spk: Optional[str] = str(spk_val) if spk_val is not None else None

            rec: Record = {
                "audio_filepath": str(audio_path),
                "text": text,
                "duration": None,
                "sample_rate": None,
                "dataset": dataset_name,
                "id": file_id,
                "speaker_id": spk,
                "language": language,
            }
            yield rec


# =========================================================================== #
# Adapters
# =========================================================================== #

# OpenSLR-53 is sharded into 10 numeric parts (0-9) plus 6 alphabetic (a-f).
_OPENSLR53_PARTS: tuple = tuple(list(range(10)) + ["a", "b", "c", "d", "e", "f"])
_OPENSLR53_BASE_URL = "https://www.openslr.org/resources/53/asr_bengali_{part}.zip"


class OpenSLR53Adapter(DatasetAdapter):
    name = "openslr53"
    language = "bn"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "OpenSLR53"
        out_dir.mkdir(parents=True, exist_ok=True)

        # utt_spk_text.tsv is a standalone top-level file (not bundled in any
        # part zip) — fetch it separately, idempotently.
        asr_bengali_dir = out_dir / "asr_bengali"
        asr_bengali_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = asr_bengali_dir / "utt_spk_text.tsv"
        if not tsv_path.exists():
            tsv_url = "https://www.openslr.org/resources/53/utt_spk_text.tsv"
            print(f"[openslr53] downloading utt_spk_text.tsv from {tsv_url}")
            urllib.request.urlretrieve(tsv_url, tsv_path)

        # 16 part zips. The zips are deleted after extraction, so their absence
        # can't signal "already done" — we drop a per-part marker instead so a
        # re-run skips the multi-GB refetch (the old code re-downloaded every
        # part on every run). A failed download/extract raises and leaves no
        # marker, so the next run retries just that part.
        for part in _OPENSLR53_PARTS:
            marker = out_dir / f".part_{part}.done"
            if marker.exists():
                continue
            zip_path = out_dir / f"asr_bengali_{part}.zip"
            url = _OPENSLR53_BASE_URL.format(part=part)
            print(f"[openslr53] downloading part {part} from {url}")
            urllib.request.urlretrieve(url, zip_path)
            print(f"[openslr53] extracting part {part}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
            zip_path.unlink(missing_ok=True)
            marker.touch()

        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        # The upstream zip lays things out as:
        #   raw_dir/asr_bengali/utt_spk_text.tsv
        #   raw_dir/asr_bengali/data/<2char>/<utt>.flac
        # Older instructions used different relative roots; support both.
        base = raw_dir / "asr_bengali"
        tsv = base / "utt_spk_text.tsv"
        data_root = base / "data"
        if not tsv.exists():
            # Some mirrors flatten one level — try raw_dir directly.
            alt_tsv = raw_dir / "utt_spk_text.tsv"
            alt_data = raw_dir / "data"
            if alt_tsv.exists():
                tsv = alt_tsv
                data_root = alt_data
            else:
                print(f"[openslr53] tsv not found under {raw_dir!s}")
                return

        with open(tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                utt_id, spk_id, text = parts[0], parts[1], parts[2]
                # OpenSLR shards utterances into 2-char prefix subfolders.
                subfolder = utt_id[:2]
                audio_path = data_root / subfolder / f"{utt_id}.flac"
                if not audio_path.exists():
                    continue
                rec: Record = {
                    "audio_filepath": str(audio_path),
                    "text": text,
                    "duration": None,
                    "sample_rate": None,
                    "dataset": self.name,
                    "id": utt_id,
                    "speaker_id": spk_id,
                    "language": self.language,
                }
                yield rec


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

        # The SDK reads MDC_API_KEY from the env. Touch it here so a missing key
        # is a hard KeyError up front (get it from your Mozilla Data Collective
        # Account -> Credentials) rather than an opaque SDK failure later.
        _ = os.environ["MDC_API_KEY"]
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


def _authenticate_kaggle() -> "object":
    from kaggle.api.kaggle_api_extended import KaggleApi

    # KaggleApi reads KAGGLE_USERNAME / KAGGLE_KEY from the env (set from .env).
    # Touch them so a missing key fails fast here rather than inside the SDK.
    _ = os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    api = KaggleApi()
    api.authenticate()
    return api


class RegSpeech12Adapter(DatasetAdapter):
    name = "regspeech12"
    language = "bn"

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


class BengaliAISpeechAdapter(DatasetAdapter):
    """Bengali.AI Speech Recognition — the Kaggle ``bengaliai-speech``
    competition train set (~1200 h of read + spontaneous Bengali speech,
    963k mp3 clips, ~26 GB).

    This is a Kaggle *competition* (not a dataset), so you must accept the
    rules once at kaggle.com/competitions/bengaliai-speech before the API
    will serve the files; otherwise the download 403s. Layout after unzip:
    ``train.csv`` (columns ``id,sentence,split``) + ``train_mp3s/<id>.mp3``
    (the unlabeled ``test_mp3s/`` is ignored). We emit every labeled row;
    the competition's own train/valid split is ignored since pack does its
    own train/val partition.
    """

    name = "bengaliai_speech"
    language = "bn"

    _COMPETITION = "bengaliai-speech"

    def download(self, dest_root: Path) -> Path:
        # Folder matches the competition slug so an existing manual download
        # (e.g. data/bengaliai-speech) is reused when --data-root points at it.
        out_dir = dest_root / self._COMPETITION
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "train.csv").exists() and (out_dir / "train_mp3s").is_dir():
            print(f"[bengaliai_speech] already extracted under {out_dir!s}")
            return out_dir

        api = _authenticate_kaggle()
        print(f"[bengaliai_speech] downloading kaggle competition {self._COMPETITION}")
        api.competition_download_files(self._COMPETITION, path=str(out_dir), quiet=False)

        zip_path = out_dir / f"{self._COMPETITION}.zip"
        if not zip_path.exists():
            zips = sorted(out_dir.glob("*.zip"))
            if not zips:
                raise FileNotFoundError(
                    f"[bengaliai_speech] no zip found in {out_dir!s} after download "
                    f"(did you accept the competition rules at "
                    f"kaggle.com/competitions/{self._COMPETITION}?)"
                )
            zip_path = zips[0]

        print(f"[bengaliai_speech] extracting {zip_path.name} ({zip_path.stat().st_size >> 30} GB)")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        zip_path.unlink(missing_ok=True)
        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        import pandas as pd

        csv = raw_dir / "train.csv"
        clips_dir = raw_dir / "train_mp3s"
        if not csv.exists():
            print(f"[bengaliai_speech] missing {csv!s}")
            return

        df = pd.read_csv(csv)
        if "id" not in df.columns or "sentence" not in df.columns:
            print(f"[bengaliai_speech] unexpected csv columns: {list(df.columns)}")
            return

        for file_id, sentence in zip(df["id"].tolist(), df["sentence"].tolist()):
            file_id = str(file_id)
            audio_path = clips_dir / f"{file_id}.mp3"
            if not audio_path.exists():
                continue
            text = None if sentence is None or pd.isna(sentence) else str(sentence)
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


class IndicVoicesAdapter(DatasetAdapter):
    name = "indicvoices"
    language = "bn"

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


class SubakKoAdapter(DatasetAdapter):
    name = "subak_ko"
    language = "bn"

    _REPO_ID = "SUST-CSE-Speech/SUBAK.KO"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "subak_ko"
        # Only fetch the parquet shards we actually use; the repo also has a
        # large unused zip under `Data/` (23.3 GB) that we don't need.
        hf_snapshot_download(
            self._REPO_ID, out_dir, allow_patterns=["data/*.parquet"]
        )
        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        yield from iter_parquet_records(
            raw_dir=raw_dir,
            dataset_name=self.name,
            language=self.language,
        )


class ShrutilipiAdapter(DatasetAdapter):
    name = "shrutilipi"
    language = "bn"

    _REPO_ID = "ai4bharat/Shrutilipi"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "shrutilipi"
        # Bengali subset lives under `bengali/` as flat train-*.parquet shards.
        hf_snapshot_download(self._REPO_ID, out_dir, allow_patterns="bengali/*")
        return out_dir

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        yield from iter_parquet_records(
            raw_dir=raw_dir,
            dataset_name=self.name,
            language=self.language,
            glob_patterns=("bengali/*.parquet", "bengali/**/*.parquet", "**/*.parquet"),
        )


class KathbathAdapter(DatasetAdapter):
    """Bengali eval/probe corpus (valid-* shards). pack does NOT special-case
    kathbath — there is no automatic routing to a probe manifest. To keep it as
    a held-out eval set, it must be excluded from the pretraining ``--datasets``
    list; that's why it's left out of the Makefile default. It remains
    registered here so it can still be downloaded explicitly via
    ``DATASETS=kathbath``.
    """

    name = "kathbath"
    language = "bn"

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


# --- registry -------------------------------------------------------------- #

REGISTRY: dict[str, type[DatasetAdapter]] = {
    "openslr53": OpenSLR53Adapter,
    "common_voice_bn": CommonVoiceBnAdapter,
    "bengaliai_speech": BengaliAISpeechAdapter,
    "regspeech12": RegSpeech12Adapter,
    "indicvoices": IndicVoicesAdapter,
    "subak_ko": SubakKoAdapter,
    "shrutilipi": ShrutilipiAdapter,
    "kathbath": KathbathAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


# =========================================================================== #
# Audit (parallel sanity check over raw audio paths before packing)
# =========================================================================== #


def _audit_one(args: tuple[int, dict[str, Any], float, float]) -> dict[str, Any]:
    """Worker: probe one audio file with soundfile.info.

    Returns a dict with at least ``index`` and ``status``. ``status`` is one of
    ``ok``, ``missing``, ``too_short``, ``too_long``, ``empty``, ``corrupt``.
    On ``ok`` the dict also carries ``duration`` (seconds, from sf.info).
    """
    import soundfile as sf

    idx, rec, min_duration, max_duration = args
    path = rec.get("audio_filepath")
    if not path or not Path(path).exists():
        return {"index": idx, "status": "missing", "path": path}
    try:
        info = sf.info(path)
    except Exception as e:
        return {"index": idx, "status": "corrupt", "path": path, "error": str(e)}
    if info.frames == 0:
        return {"index": idx, "status": "empty", "path": path}
    dur = float(info.duration)
    if dur < min_duration:
        return {"index": idx, "status": "too_short", "path": path, "duration": dur}
    if dur > max_duration:
        return {"index": idx, "status": "too_long", "path": path, "duration": dur}
    return {"index": idx, "status": "ok", "path": path, "duration": dur}


def audit_records(
    records: Iterable[Record],
    num_workers: int = 4,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> tuple[list[Record], dict[str, Any]]:
    """Probe every record's audio file in parallel and drop bad rows.

    Returns ``(kept_records, report)``. The kept records have their ``duration``
    field overwritten with the measured value. The report dict has per-status
    counts plus the parameters used.
    """
    from tqdm import tqdm

    rec_list: list[Record] = list(records)
    work = [
        (i, dict(r), min_duration, max_duration) for i, r in enumerate(rec_list)
    ]

    results: list[dict[str, Any]] = []
    if num_workers <= 1:
        for w in tqdm(work, total=len(work), desc="audit"):
            results.append(_audit_one(w))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for res in tqdm(
                ex.map(_audit_one, work, chunksize=64),
                total=len(work),
                desc="audit",
            ):
                results.append(res)

    counts: dict[str, int] = {}
    kept: list[Record] = []
    for res in results:
        st = res["status"]
        counts[st] = counts.get(st, 0) + 1
        if st == "ok":
            rec = rec_list[res["index"]]
            rec["duration"] = res["duration"]
            kept.append(rec)

    report = {
        "total": len(rec_list),
        "kept": len(kept),
        "counts": counts,
        "min_duration": min_duration,
        "max_duration": max_duration,
    }
    print("[audit] summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"[audit] kept {len(kept)} / {len(rec_list)} rows")
    return kept, report


# =========================================================================== #
# Pack (resample + transcode + split + manifest emission)
# =========================================================================== #


def _safe_id(rec: Record) -> str:
    """Stable filename stem: prefer ``id``, fall back to a hash of the path."""
    rid = rec.get("id")
    if rid:
        # Sanitize: replace path separators / spaces to keep the filename safe.
        return str(rid).replace("/", "_").replace("\\", "_").replace(" ", "_")
    src = rec.get("audio_filepath", "")
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]


def _transcode_one(
    rec: Record,
    staging_dir: Path,
    target_sr: int,
    min_duration: float,
    max_duration: float,
    skip_existing: bool,
) -> tuple[Record | None, str]:
    """Validate + transcode one source file in a single read of the file.

    Reads the source once, filters on duration, and writes
    ``staging_dir/audio/<dataset>/<id>.flac``. Returns ``(record, status)``
    where ``status`` is one of ``ok``, ``missing``, ``decode_error``,
    ``too_short``, ``too_long``, ``transcode_error``. The record is ``None``
    for every non-``ok`` status. This is the only place that drops bad rows —
    there is no separate pre-pass.
    """
    import soundfile as sf
    import torchaudio
    import torchaudio.functional as AF

    src = rec.get("audio_filepath")
    if not src or not Path(src).exists():
        return None, "missing"

    dataset = rec["dataset"]
    stem = _safe_id(rec)
    rel_path = Path("audio") / dataset / f"{stem}.flac"
    out_path = staging_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_rec: Record = dict(rec)  # shallow copy; we don't mutate caller's record

    if skip_existing and out_path.exists():
        # Trust a prior transcode. Refresh duration from the existing file so
        # the manifest is internally consistent.
        try:
            info = sf.info(str(out_path))
        except Exception:
            out_path.unlink(missing_ok=True)  # cached file is bad — redo it
        else:
            new_rec["audio_filepath"] = rel_path.as_posix()
            new_rec["duration"] = float(info.duration)
            new_rec["sample_rate"] = int(info.samplerate)
            return new_rec, "ok"

    try:
        wav, sr = torchaudio.load(src)
    except Exception as e:
        print(f"[pack] decode failed for {src}: {e}")
        return None, "decode_error"

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    duration = wav.size(-1) / float(sr)
    if duration < min_duration:
        return None, "too_short"
    if duration > max_duration:
        return None, "too_long"

    try:
        if int(sr) != int(target_sr):
            wav = AF.resample(wav, int(sr), int(target_sr))
        samples = wav.squeeze(0).contiguous().cpu().numpy()
        sf.write(
            str(out_path), samples, int(target_sr), format="FLAC", subtype="PCM_16"
        )
    except Exception as e:
        print(f"[pack] transcode failed for {src}: {e}")
        return None, "transcode_error"

    new_rec["audio_filepath"] = rel_path.as_posix()
    new_rec["sample_rate"] = int(target_sr)
    new_rec["duration"] = float(samples.shape[-1] / target_sr)
    return new_rec, "ok"


def _per_dataset_split(
    records: list[Record], val_pct: float, rng: random.Random
) -> tuple[list[Record], list[Record]]:
    """Split per ``dataset`` so every source contributes to both splits."""
    by_ds: dict[str, list[Record]] = {}
    for r in records:
        by_ds.setdefault(r["dataset"], []).append(r)

    train: list[Record] = []
    val: list[Record] = []
    for ds, rows in by_ds.items():
        rng.shuffle(rows)
        n = len(rows)
        if n == 0:
            continue
        n_val = max(1, int(round(n * val_pct))) if n > 1 else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val


def _write_jsonl(path: Path, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_readme(staging_dir: Path, build_meta: dict[str, Any]) -> None:
    sources = ", ".join(build_meta.get("adapters", []))
    total = build_meta.get("packed_total", 0)
    body = (
        "---\n"
        "language:\n"
        "- bn\n"
        "license: other\n"
        "task_categories:\n"
        "- automatic-speech-recognition\n"
        "tags:\n"
        "- bengali\n"
        "- audio\n"
        "- self-supervised\n"
        "---\n\n"
        "# Bengali Speech Corpus\n\n"
        f"Aggregated Bengali speech corpus packed by `scripts/housekeeping.py`.\n\n"
        f"- Sources: {sources}\n"
        f"- Total rows (packed): {total}\n"
        f"- Sample rate: {build_meta.get('target_sr')} Hz mono FLAC\n\n"
        "Manifests under `manifests/`. Audio paths are relative to the repo root.\n"
        "License placeholder: each underlying source retains its original license; verify before redistribution.\n"
    )
    (staging_dir / "README.md").write_text(body, encoding="utf-8")


def _write_build_meta(staging_dir: Path, build_meta: dict[str, Any]) -> None:
    import yaml

    (staging_dir / "build_meta.yaml").write_text(
        yaml.safe_dump(build_meta, sort_keys=False), encoding="utf-8"
    )


def pack_to_dir(
    adapters: list[DatasetAdapter],
    download_root: Path,
    staging_dir: Path,
    target_sr: int = 16000,
    val_pct: float = 0.05,
    asr_probe_pct: float = 0.2,
    seed: int = 42,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    skip_existing: bool = True,
    asr_probe_max_rows: int = 50000,
    asr_probe_val_max_rows: int = 5000,
) -> dict[str, Any]:
    """Download (idempotent), transcode to 16k mono FLAC (with inline duration
    filtering), split, write manifests.

    Returns the build_meta dict (also written to ``staging_dir/build_meta.yaml``).
    """
    from tqdm import tqdm


    staging_dir = Path(staging_dir)
    download_root = Path(download_root)
    staging_dir.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    # 1) Download + iter_records per adapter.
    raw_counts: dict[str, int] = {}
    all_records: list[Record] = []
    for adapter in adapters:
        print(f"[pack] downloading {adapter.name} -> {download_root}")
        raw_dir = adapter.download(download_root)
        print(f"[pack] iterating records for {adapter.name} from {raw_dir}")
        recs = list(adapter.iter_records(Path(raw_dir)))
        for r in recs:
            r["dataset"] = adapter.name  # belt-and-suspenders
        raw_counts[adapter.name] = len(recs)
        all_records.extend(recs)
        print(f"[pack]   {adapter.name}: {len(recs)} raw records")

    # 2) Single validate + transcode pass. Each file is read once: duration
    # filtering and bad-row dropping happen inside _transcode_one (no separate
    # audit pre-pass). Serial because torchaudio/soundfile are I/O-bound and
    # parallel decode adds memory + process-startup overhead that doesn't pay
    # off until well above 100k files.
    packed: list[Record] = []
    status_counts: dict[str, int] = {}
    for rec in tqdm(all_records, total=len(all_records), desc="transcode"):
        new_rec, status = _transcode_one(
            rec, staging_dir, target_sr, min_duration, max_duration, skip_existing
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        if new_rec is not None:
            packed.append(new_rec)

    print("[pack] transcode summary:")
    for k, v in sorted(status_counts.items()):
        print(f"  {k}: {v}")
    print(f"[pack] kept {len(packed)} / {len(all_records)} rows")

    packed_counts: dict[str, int] = {}
    for r in packed:
        packed_counts[r["dataset"]] = packed_counts.get(r["dataset"], 0) + 1

    # 4) Deterministic shuffle.
    rng = random.Random(seed)
    rng.shuffle(packed)

    # 5) Per-dataset train/val split.
    train_rows, val_rows = _per_dataset_split(packed, val_pct, rng)
    print(f"[pack] split: {len(train_rows)} train / {len(val_rows)} val")

    # 6) Write the four manifests.
    manifests_dir = staging_dir / "manifests"
    _write_jsonl(manifests_dir / "train.jsonl", train_rows)
    _write_jsonl(manifests_dir / "val.jsonl", val_rows)

    train_text = [r for r in train_rows if r.get("text")]
    val_text = [r for r in val_rows if r.get("text")]

    probe_train_n = min(
        asr_probe_max_rows, max(0, int(round(len(train_text) * asr_probe_pct)))
    )
    probe_val_n = min(asr_probe_val_max_rows, len(val_text))

    # Stable subsample via a dedicated RNG so train.jsonl ordering doesn't leak in.
    probe_rng = random.Random(seed + 1)
    probe_train = list(train_text)
    probe_rng.shuffle(probe_train)
    probe_train = probe_train[:probe_train_n]

    probe_val = list(val_text)
    probe_rng.shuffle(probe_val)
    probe_val = probe_val[:probe_val_n]

    _write_jsonl(manifests_dir / "asr_probe_train.jsonl", probe_train)
    _write_jsonl(manifests_dir / "asr_probe_val.jsonl", probe_val)
    print(
        f"[pack] asr probe: {len(probe_train)} train / {len(probe_val)} val "
        f"(text-labeled pool: {len(train_text)} train / {len(val_text)} val)"
    )

    # 7) build_meta.yaml.
    build_meta: dict[str, Any] = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "adapters": [a.name for a in adapters],
        "target_sr": int(target_sr),
        "val_pct": float(val_pct),
        "seed": int(seed),
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "raw_counts": raw_counts,
        "packed_counts": packed_counts,
        "transcode_status_counts": status_counts,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "asr_probe_train_rows": len(probe_train),
        "asr_probe_val_rows": len(probe_val),
        "packed_total": len(packed),
    }
    _write_build_meta(staging_dir, build_meta)

    # 8) Minimal README dataset card.
    _write_readme(staging_dir, build_meta)

    print(f"[pack] done. staging dir: {staging_dir}")
    return build_meta


# =========================================================================== #
# Push (upload a packed staging dir to a HF dataset repo)
# =========================================================================== #


_GITATTRIBUTES = (
    "*.flac filter=lfs diff=lfs merge=lfs -text\n"
    "*.wav filter=lfs diff=lfs merge=lfs -text\n"
    "*.mp3 filter=lfs diff=lfs merge=lfs -text\n"
)


def push_to_hub(
    staging_dir: Path,
    repo_id: str,
    commit_message: str | None = None,
    private: bool = True,
) -> str:
    """Create-or-update a HF dataset repo from ``staging_dir``. Returns the repo URL."""
    from huggingface_hub import HfApi


    staging_dir = Path(staging_dir)
    tok = os.environ["HF_TOKEN"]

    # Make LFS tracking explicit for the binary audio extensions.
    gitattributes = staging_dir / ".gitattributes"
    if not gitattributes.exists():
        gitattributes.write_text(_GITATTRIBUTES, encoding="utf-8")

    api = HfApi(token=tok)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )

    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    msg = commit_message or f"pack {git_hash}"
    print(f"[push] uploading {staging_dir} -> {repo_id} ({msg})")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=msg,
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[push] done: {url}")
    return url


# =========================================================================== #
# Fetch (pull a packed HF dataset repo onto local disk for training)
# =========================================================================== #


def fetch_dataset(
    repo_id: str,
    dest: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Snapshot-download a packed dataset repo. Returns the local repo root."""
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] downloading {repo_id} -> {dest}")
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        token=os.environ["HF_TOKEN"],
        allow_patterns=allow_patterns,
    )
    local_root = Path(local_root)

    manifests_dir = local_root / "manifests"
    if manifests_dir.exists():
        print("[fetch] resolved manifest paths:")
        for jp in sorted(manifests_dir.glob("*.jsonl")):
            print(f"  {jp}")
    else:
        print(f"[fetch] note: no manifests/ subdir under {local_root}")
    return local_root


# =========================================================================== #
# Publish checkpoint (upload last.pt + model card to a HF model repo)
# =========================================================================== #


def _render_model_card(repo_id: str, step: object, cfg_yaml: str) -> str:
    """Compose a minimal HF model card README with YAML front matter."""
    wandb_project = os.environ.get("WANDB_PROJECT", "")
    wandb_line = f"- W&B project: `{wandb_project}`\n" if wandb_project else ""
    return (
        "---\n"
        "language:\n"
        "- bn\n"
        "license: other\n"
        "library_name: pytorch\n"
        "tags:\n"
        "- audio\n"
        "- self-supervised\n"
        "- bengali\n"
        "- conformer\n"
        "- lejepa\n"
        "---\n\n"
        f"# {repo_id}\n\n"
        "Continuous latent autoencoder for Bengali speech.\n\n"
        "## Architecture\n\n"
        "- Encoder: Conformer\n"
        "- Self-supervised objective: LeJEPA\n"
        "- Reconstruction loss: multi-resolution STFT\n\n"
        "## Training\n\n"
        f"- Step: `{step}`\n"
        f"{wandb_line}"
        "\n"
        "## Config\n\n"
        "```yaml\n"
        f"{cfg_yaml}"
        "```\n\n"
        "## How to load\n\n"
        "```python\n"
        "import torch\n"
        "ckpt = torch.load('last.pt', map_location='cpu')\n"
        "state_dict = ckpt['model']\n"
        "cfg = ckpt['cfg']\n"
        "```\n"
    )


def publish_checkpoint(
    ckpt_path: Path,
    repo_id: str,
    extra_files: list[Path] | None = None,
    commit_message: str | None = None,
    private: bool = True,
) -> str:
    """Push ``last.pt`` + a generated model card + ``config.yaml`` to a HF model repo."""
    import torch
    import yaml
    from huggingface_hub import HfApi


    ckpt_path = Path(ckpt_path)
    tok = os.environ["HF_TOKEN"]

    api = HfApi(token=tok)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=private,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    step = ckpt.get("step", "?")
    cfg = ckpt.get("cfg", {})
    cfg_yaml = yaml.safe_dump(cfg, sort_keys=False)

    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    msg = commit_message or f"publish step={step} git={git_hash}"
    print(f"[publish] uploading {ckpt_path} -> {repo_id} ({msg})")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        readme_path = tmp_dir / "README.md"
        readme_path.write_text(
            _render_model_card(repo_id, step, cfg_yaml), encoding="utf-8"
        )
        config_path = tmp_dir / "config.yaml"
        config_path.write_text(cfg_yaml, encoding="utf-8")

        # Upload ckpt under its canonical name.
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo="last.pt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
        )
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
        )
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.yaml",
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
        )
        for ef in extra_files or []:
            ef = Path(ef)
            api.upload_file(
                path_or_fileobj=str(ef),
                path_in_repo=ef.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=msg,
            )

    url = f"https://huggingface.co/{repo_id}"
    print(f"[publish] done: {url}")
    return url


# =========================================================================== #
# CLI
# =========================================================================== #
#
# Tokens and repo/path config come straight from the environment (sourced from
# .env by setup.sh). Secrets are read at point of use, so a missing one is a
# hard KeyError, not a silent fallback. Repo IDs and the data root are not
# secrets, so they get sensible literal defaults.

_DEFAULT_HF_REPO = "aryanrahman/clae-bengali"
_DEFAULT_CKPT_REPO = "aryanrahman/clae-bengali-encoder"


def _data_root(arg: str | None) -> Path:
    # Default: a gitignored `datasets/` folder at the repo root, created on demand.
    return Path(arg or os.environ.get("DATA_ROOT") or (_REPO_ROOT / "datasets"))


def _parse_datasets(s: str | None) -> List[str]:
    """Comma-separated list of adapter names; empty/None -> all registered."""
    if not s:
        return sorted(REGISTRY)
    out = [x.strip() for x in s.split(",") if x.strip()]
    for name in out:
        if name not in REGISTRY:
            raise SystemExit(f"Unknown dataset {name!r}. Available: {sorted(REGISTRY)}")
    return out


class _LimitedAdapter(DatasetAdapter):
    """Wrap an adapter to cap ``iter_records`` at ``limit`` rows (smoke testing)."""

    def __init__(self, inner: DatasetAdapter, limit: int) -> None:
        self._inner = inner
        self._limit = int(limit)
        self.name = inner.name
        self.language = inner.language

    def download(self, dest_root: Path) -> Path:
        return self._inner.download(dest_root)

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        return itertools.islice(self._inner.iter_records(raw_dir), self._limit)


def _build_adapters(names: List[str], limit: int | None) -> List[DatasetAdapter]:
    adapters: List[DatasetAdapter] = [get_adapter(n) for n in names]
    if limit is not None and limit > 0:
        adapters = [_LimitedAdapter(a, limit) for a in adapters]
    return adapters


# --- download -------------------------------------------------------------- #


def _add_download(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated adapter names. Default: all registered.",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Root for raw archives. Default: $DATA_ROOT env.",
    )
    p.set_defaults(func=_run_download)


def _run_download(args: argparse.Namespace) -> None:
    names = _parse_datasets(args.datasets)
    root = _data_root(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        adapter = get_adapter(name)
        print(f"[housekeeping] download: {name} -> {root}")
        adapter.download(root)


# --- audit ----------------------------------------------------------------- #


def _add_audit(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--staging-dir",
        required=True,
        help="Staging dir containing a manifests/ subdir of JSONL files.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.set_defaults(func=_run_audit)


def _run_audit(args: argparse.Namespace) -> None:
    """Standalone audit: probe every JSONL under ``<staging-dir>/manifests/``.

    Debug-only. ``pack_to_dir`` validates + filters inline while transcoding, so
    a normal ``build`` does not need this. Use it to verify a packed manifest
    after the fact, e.g. on a different machine, without re-running the pack.
    """
    staging = Path(args.staging_dir)
    manifests_dir = staging / "manifests"
    if not manifests_dir.is_dir():
        raise SystemExit(f"[housekeeping] no manifests/ under {staging}")

    for jp in sorted(manifests_dir.glob("*.jsonl")):
        print(f"[housekeeping] audit: {jp}")
        with open(jp, "r", encoding="utf-8") as f:
            records: list[Record] = [json.loads(line) for line in f if line.strip()]
        # Manifests written by pack store paths relative to the staging dir
        # root. Absolutize so the audit worker's existence check is accurate
        # regardless of cwd.
        for r in records:
            ap = r.get("audio_filepath")
            if ap and not os.path.isabs(ap):
                r["audio_filepath"] = str(staging / ap)
        audit_records(
            records,
            num_workers=args.num_workers,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )


# --- build ----------------------------------------------------------------- #


def _add_build(p: argparse.ArgumentParser) -> None:
    p.add_argument("--datasets", default=None)
    p.add_argument(
        "--staging-dir",
        required=True,
        help="Output directory: receives audio/, manifests/, README.md, build_meta.yaml.",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Root used by adapters for raw archives. Default: $DATA_ROOT env.",
    )
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--val-pct", type=float, default=0.05)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap rows per adapter (smoke testing). Default: no cap.",
    )
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_run_build)


def _run_build(args: argparse.Namespace) -> None:
    names = _parse_datasets(args.datasets)
    adapters = _build_adapters(names, args.limit)
    print(f"[housekeeping] build: {names} -> {args.staging_dir}")
    pack_to_dir(
        adapters=adapters,
        download_root=_data_root(args.data_root),
        staging_dir=Path(args.staging_dir),
        target_sr=args.target_sr,
        val_pct=args.val_pct,
        seed=args.seed,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )


# --- push ------------------------------------------------------------------ #


def _add_push(p: argparse.ArgumentParser) -> None:
    p.add_argument("--staging-dir", required=True)
    p.add_argument("--repo-id", default=None, help="Default: $HF_DATASET_REPO env.")
    p.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public (default: private).",
    )
    p.add_argument("--commit-message", default=None)
    p.set_defaults(func=_run_push)


def _run_push(args: argparse.Namespace) -> None:
    repo_id = args.repo_id or os.environ.get("HF_DATASET_REPO", _DEFAULT_HF_REPO)
    staging = Path(args.staging_dir)
    # Quick row count for the progress message.
    n_files = sum(1 for _ in staging.rglob("*") if _.is_file())
    print(f"[housekeeping] push: {n_files:,} files -> {repo_id}")
    push_to_hub(
        staging_dir=staging,
        repo_id=repo_id,
        commit_message=args.commit_message,
        private=not args.public,
    )


# --- fetch ----------------------------------------------------------------- #


def _add_fetch(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-id", default=None, help="Default: $HF_DATASET_REPO env.")
    p.add_argument(
        "--dest",
        default=None,
        help="Local destination. Default: $DATA_ROOT env.",
    )
    p.set_defaults(func=_run_fetch)


def _run_fetch(args: argparse.Namespace) -> None:
    repo_id = args.repo_id or os.environ.get("HF_DATASET_REPO", _DEFAULT_HF_REPO)
    dest = _data_root(args.dest)
    print(f"[housekeeping] fetch: {repo_id} -> {dest}")
    fetch_dataset(repo_id=repo_id, dest=dest)


# --- publish-checkpoint ---------------------------------------------------- #


def _add_publish_checkpoint(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", required=True, help="Path to last.pt")
    p.add_argument("--repo-id", default=None, help="Default: $HF_MODEL_REPO env.")
    p.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public (default: private).",
    )
    p.add_argument("--commit-message", default=None)
    p.set_defaults(func=_run_publish_checkpoint)


def _run_publish_checkpoint(args: argparse.Namespace) -> None:
    repo_id = args.repo_id or os.environ.get("HF_MODEL_REPO", _DEFAULT_CKPT_REPO)
    print(f"[housekeeping] publish-checkpoint: {args.ckpt} -> {repo_id}")
    publish_checkpoint(
        ckpt_path=Path(args.ckpt),
        repo_id=repo_id,
        commit_message=args.commit_message,
        private=not args.public,
    )


# --- pack-and-push (convenience) ------------------------------------------- #


def _add_pack_and_push(p: argparse.ArgumentParser) -> None:
    p.add_argument("--datasets", default=None)
    p.add_argument("--repo-id", default=None, help="Default: $HF_DATASET_REPO env.")
    p.add_argument(
        "--data-root",
        default=None,
        help="Root used by adapters for raw archives. Default: $DATA_ROOT env.",
    )
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--val-pct", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--staging-dir",
        default=None,
        help="If unset, uses a tmp dir. Set this to keep artifacts (implies --keep-staging).",
    )
    p.add_argument(
        "--keep-staging",
        action="store_true",
        help="When --staging-dir is unset, do not delete the tmp staging dir on exit.",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public (default: private).",
    )
    p.add_argument("--commit-message", default=None)
    p.set_defaults(func=_run_pack_and_push)


def _run_pack_and_push(args: argparse.Namespace) -> None:
    names = _parse_datasets(args.datasets)
    adapters = _build_adapters(names, args.limit)
    repo_id = args.repo_id or os.environ.get("HF_DATASET_REPO", _DEFAULT_HF_REPO)
    data_root = _data_root(args.data_root)

    def _do(staging: Path) -> None:
        print(f"[housekeeping] pack-and-push: build {names} -> {staging}")
        pack_to_dir(
            adapters=adapters,
            download_root=data_root,
            staging_dir=staging,
            target_sr=args.target_sr,
            val_pct=args.val_pct,
            seed=args.seed,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        n_files = sum(1 for _ in staging.rglob("*") if _.is_file())
        print(f"[housekeeping] pack-and-push: push {n_files:,} files -> {repo_id}")
        push_to_hub(
            staging_dir=staging,
            repo_id=repo_id,
            commit_message=args.commit_message,
            private=not args.public,
        )

    if args.staging_dir:
        staging = Path(args.staging_dir)
        staging.mkdir(parents=True, exist_ok=True)
        _do(staging)
    elif args.keep_staging:
        staging = Path(tempfile.mkdtemp(prefix="pack_"))
        print(f"[housekeeping] pack-and-push: --keep-staging set, using {staging}")
        _do(staging)
    else:
        with tempfile.TemporaryDirectory(prefix="pack_") as tmp:
            _do(Path(tmp))


# --- dispatch -------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(prog="housekeeping.py")
    sub = ap.add_subparsers(dest="command", required=True)

    _add_download(sub.add_parser("download", help="Download raw archives."))
    _add_audit(
        sub.add_parser(
            "audit", help="Probe staging manifests (debug; build runs audit internally)."
        )
    )
    _add_build(sub.add_parser("build", help="Pack records into a staging dir."))
    _add_push(sub.add_parser("push", help="Upload a staging dir to HF Hub."))
    _add_fetch(sub.add_parser("fetch", help="Snapshot-download a packed dataset repo."))
    _add_publish_checkpoint(
        sub.add_parser("publish-checkpoint", help="Upload a checkpoint to HF Hub.")
    )
    _add_pack_and_push(
        sub.add_parser("pack-and-push", help="Build + push in one shot.")
    )

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
