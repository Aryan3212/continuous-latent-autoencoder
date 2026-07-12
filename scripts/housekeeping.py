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
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence, TypedDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class Record(TypedDict, total=False):
    audio_filepath: str
    text: Optional[str]
    duration: Optional[float]
    sample_rate: Optional[int]
    dataset: str
    id: Optional[str]
    speaker_id: Optional[str]
    language: Optional[str]


class DatasetAdapter(abc.ABC):
    name: str
    language: str = "bn"

    @abc.abstractmethod
    def download(self, dest_root: Path) -> Path:
        ...

    @abc.abstractmethod
    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        ...


# =========================================================================== #
# Shared helper: HF datasets distributed as parquet with inline audio bytes
# (used by IndicVoices, SUBAK.KO, Shrutilipi, Kathbath)
# =========================================================================== #


def hf_snapshot_download(
    repo_id: str,
    dest_dir: Path,
    allow_patterns: Optional[Sequence[str] | str] = None,
) -> Path:
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)

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
    import pyarrow.parquet as pq

    extract_root = raw_dir / extract_subdir
    extract_root.mkdir(parents=True, exist_ok=True)

    parquet_files: list[Path] = []
    for pattern in glob_patterns:
        parquet_files.extend(sorted(raw_dir.glob(pattern)))
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

            ext = ".flac"
            if inline_path:
                guess = Path(inline_path).suffix.lower()
                if guess in (".flac", ".wav", ".mp3", ".ogg", ".opus"):
                    ext = guess

            audio_path = extract_root / f"{file_id}{ext}"
            if not audio_path.exists():
                if audio_bytes is None:
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

_OPENSLR53_PARTS: tuple = tuple(list(range(10)) + ["a", "b", "c", "d", "e", "f"])
_OPENSLR53_BASE_URL = "https://www.openslr.org/resources/53/asr_bengali_{part}.zip"


class OpenSLR53Adapter(DatasetAdapter):
    name = "openslr53"
    language = "bn"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "OpenSLR53"
        out_dir.mkdir(parents=True, exist_ok=True)

        asr_bengali_dir = out_dir / "asr_bengali"
        asr_bengali_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = asr_bengali_dir / "utt_spk_text.tsv"
        if not tsv_path.exists():
            tsv_url = "https://www.openslr.org/resources/53/utt_spk_text.tsv"
            print(f"[openslr53] downloading utt_spk_text.tsv from {tsv_url}")
            urllib.request.urlretrieve(tsv_url, tsv_path)

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
        base = raw_dir / "asr_bengali"
        tsv = base / "utt_spk_text.tsv"
        data_root = base / "data"
        if not tsv.exists():
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
    name = "common_voice_bn"
    language = "bn"

    _DATASET_ID = "cmqim44fo00tinr07mbu70eg7"
    _API_BASES = (
        "https://mozilladatacollective.com/api",
        "https://dev.mozilladatacollective.com/api",
    )

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "common_voice_bn"
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._find_validated_tsv(out_dir) is not None:
            print(f"[common_voice_bn] already extracted under {out_dir!s}")
            return out_dir

        import requests

        api_key = os.environ["MDC_API_KEY"]
        headers = {"Authorization": f"Bearer {api_key}"}
        bases = (
            [os.environ["MDC_API_BASE"]]
            if os.environ.get("MDC_API_BASE")
            else list(self._API_BASES)
        )

        info = None
        last_err: Exception | None = None
        for base in bases:
            url = f"{base}/datasets/{self._DATASET_ID}/download"
            try:
                print(f"[common_voice_bn] requesting presigned URL: {url}")
                resp = requests.post(url, headers=headers, timeout=60)
                resp.raise_for_status()
                info = resp.json()
                break
            except Exception as e:
                last_err = e
                print(f"[common_voice_bn] {url} failed: {e}")
        if info is None:
            raise RuntimeError(
                "MDC download request failed on all hosts "
                f"(did you accept the dataset terms in the web UI?): {last_err}"
            )

        dl_url = info["downloadUrl"]
        filename = info.get("filename") or "common_voice_bn.tar.gz"
        size_gb = int(info.get("sizeBytes") or 0) >> 30
        tar_path = out_dir / filename

        print(f"[common_voice_bn] downloading {filename} (~{size_gb} GB)")
        with requests.get(dl_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

        print(f"[common_voice_bn] extracting {tar_path.name}")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(out_dir)

        return out_dir

    @staticmethod
    def _find_validated_tsv(raw_dir: Path) -> Path | None:
        for name in ("validated.tsv", "train.tsv"):
            hits = sorted(raw_dir.rglob(name))
            if hits:
                return hits[0]
        return None

    @staticmethod
    def _find_clips_dir(raw_dir: Path) -> Path | None:
        direct = raw_dir / "clips"
        if direct.is_dir():
            return direct
        for pat in ("*/clips", "*/*/clips", "*/*/*/clips"):
            for p in raw_dir.glob(pat):
                if p.is_dir():
                    return p
        hits = [p for p in raw_dir.rglob("clips") if p.is_dir()]
        return hits[0] if hits else None

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        import pandas as pd

        clips_dir = self._find_clips_dir(raw_dir)
        if clips_dir is None:
            print(f"[common_voice_bn] no clips/ dir under {raw_dir!s}")
            return
        cv_dir = clips_dir.parent

        meta: dict[str, tuple[Optional[str], Optional[str]]] = {}
        for name in ("validated.tsv", "invalidated.tsv", "other.tsv"):
            tsv = cv_dir / name
            if not tsv.exists():
                continue
            try:
                df = pd.read_csv(
                    tsv,
                    sep="\t",
                    low_memory=False,
                    usecols=lambda c: c in ("path", "sentence", "client_id"),
                )
            except Exception as e:
                print(f"[common_voice_bn] failed to read {tsv.name}: {e}")
                continue
            if "path" not in df.columns:
                print(f"[common_voice_bn] {tsv.name}: no 'path' column; skipping")
                continue
            n = len(df)
            paths = df["path"].tolist()
            sents = df["sentence"].tolist() if "sentence" in df.columns else [None] * n
            clients = df["client_id"].tolist() if "client_id" in df.columns else [None] * n
            for fname, sentence, client in zip(paths, sents, clients):
                if fname is None or (isinstance(fname, float) and pd.isna(fname)):
                    continue
                fname = str(fname)
                if fname in meta:
                    continue
                text = (
                    None
                    if sentence is None or (isinstance(sentence, float) and pd.isna(sentence))
                    else str(sentence)
                )
                spk = (
                    None
                    if client is None or (isinstance(client, float) and pd.isna(client))
                    else str(client)
                )
                meta[fname] = (text, spk)

        emitted = 0
        with os.scandir(clips_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                fname = entry.name
                text, spk = meta.get(fname, (None, None))
                emitted += 1
                yield {
                    "audio_filepath": entry.path,
                    "text": text,
                    "duration": None,
                    "sample_rate": None,
                    "dataset": self.name,
                    "id": Path(fname).stem,
                    "speaker_id": spk,
                    "language": self.language,
                }
        if emitted == 0:
            print(f"[common_voice_bn] clips/ dir is empty: {clips_dir!s}")


def _authenticate_kaggle() -> "object":
    from kaggle.api.kaggle_api_extended import KaggleApi

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
    name = "bengaliai_speech"
    language = "bn"

    _COMPETITION = "bengaliai-speech"

    def download(self, dest_root: Path) -> Path:
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

        for _, row in df.iterrows():
            file_id = str(row["id"])
            audio_path = clips_dir / f"{file_id}.mp3"
            if not audio_path.exists():
                continue
            sentence = row["sentence"]
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
    name = "kathbath"
    language = "bn"

    _REPO_ID = "ai4bharat/Kathbath"

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "kathbath"
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


def _per_dataset_split(
    records: list[Record], val_pct: float, rng: random.Random
) -> tuple[list[Record], list[Record]]:
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


def build_manifests_only(
    adapters_with_dirs: list[tuple[DatasetAdapter, Path]],
    out_dir: Path,
    val_pct: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_counts: dict[str, int] = {}
    all_records: list[Record] = []
    for adapter, raw_dir in adapters_with_dirs:
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise SystemExit(
                f"[manifest] raw dir for {adapter.name} does not exist: {raw_dir}"
            )
        print(f"[manifest] {adapter.name}: iterating records from {raw_dir}")
        recs = list(adapter.iter_records(raw_dir))
        for r in recs:
            r["dataset"] = adapter.name
            p = r.get("audio_filepath")
            if p and not os.path.isabs(p):
                r["audio_filepath"] = str((raw_dir / p).resolve())
        raw_counts[adapter.name] = len(recs)
        all_records.extend(recs)
        print(f"[manifest]   {adapter.name}: {len(recs)} records")
        if not recs:
            listing = (
                sorted(p.name for p in raw_dir.iterdir())[:25]
                if raw_dir.is_dir()
                else []
            )
            print(
                f"[manifest]   WARNING: 0 records for {adapter.name}. "
                f"Top-level of {raw_dir}: {listing}"
            )

    if not all_records:
        raise SystemExit("[manifest] no records from any adapter — check --map paths.")

    rng = random.Random(seed)
    rng.shuffle(all_records)
    train_rows, val_rows = _per_dataset_split(all_records, val_pct, rng)
    print(f"[manifest] split: {len(train_rows)} train / {len(val_rows)} val")

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)

    meta: dict[str, Any] = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "mode": "manifest_only",
        "sources": {a.name: str(d) for a, d in adapters_with_dirs},
        "raw_counts": raw_counts,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_pct": float(val_pct),
        "seed": int(seed),
    }
    import yaml

    (out_dir / "build_meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False), encoding="utf-8"
    )
    print(f"[manifest] done. manifests in: {out_dir}")
    return meta


# =========================================================================== #
# CLI
# =========================================================================== #

_DEFAULT_HF_REPO = "aryanrahman/clae-bengali"
_DEFAULT_CKPT_REPO = "aryanrahman/clae-bengali-encoder"


def _data_root(arg: str | None) -> Path:
    return Path(arg or os.environ.get("DATA_ROOT") or (_REPO_ROOT / "datasets"))


def _parse_datasets(s: str | None) -> List[str]:
    if not s:
        return sorted(REGISTRY)
    out = [x.strip() for x in s.split(",") if x.strip()]
    for name in out:
        if name not in REGISTRY:
            raise SystemExit(f"Unknown dataset {name!r}. Available: {sorted(REGISTRY)}")
    return out


class _LimitedAdapter(DatasetAdapter):
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


# --- make-manifests (manifest-only build over attached raw datasets) ------- #


def _add_make_manifests(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Adapter name = its mounted raw dir. Repeatable. e.g. "
        "--map regspeech12=/kaggle/input/regspeech12 "
        "--map common_voice_bn=/kaggle/input/common-voice-24-bn",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Root for downloaded raw archives (used with --datasets instead of --map).",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated adapter names (used with --data-root). Default: all registered.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Where to write train.jsonl / val.jsonl (e.g. /kaggle/working/manifests).",
    )
    p.add_argument("--val-pct", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_run_make_manifests)


def _run_make_manifests(args: argparse.Namespace) -> None:
    if args.data_root:
        names = _parse_datasets(args.datasets)
        root = Path(args.data_root)
        pairs: list[tuple[DatasetAdapter, Path]] = []
        for name in names:
            adapter = get_adapter(name)
            raw_dir = adapter.download(root)
            if not raw_dir.exists():
                raise SystemExit(f"[manifest] raw dir for {name} does not exist: {raw_dir}")
            pairs.append((adapter, raw_dir))
    elif args.map:
        pairs = []
        for m in args.map:
            if "=" not in m:
                raise SystemExit(f"--map must be NAME=PATH, got: {m!r}")
            name, path = m.split("=", 1)
            pairs.append((get_adapter(name), Path(path)))
    else:
        raise SystemExit("Either --data-root or --map NAME=PATH is required.")
    print(f"[housekeeping] make-manifests: {[a.name for a, _ in pairs]} -> {args.out_dir}")
    build_manifests_only(
        adapters_with_dirs=pairs,
        out_dir=Path(args.out_dir),
        val_pct=args.val_pct,
        seed=args.seed,
    )


# --- fetch-checkpoint ------------------------------------------------------ #


def fetch_checkpoint(
    repo_id: str, dest: Path, filename: str = "last.pt"
) -> Optional[Path]:
    from huggingface_hub import hf_hub_download

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception:
        print(f"[fetch-ckpt] {repo_id}:{filename} unavailable — starting fresh.")
        return None

    import shutil

    shutil.copyfile(local, dest)
    print(f"[fetch-ckpt] {repo_id}:{filename} -> {dest}")
    return dest


def _add_fetch_checkpoint(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-id", default=None, help="Default: $HF_MODEL_REPO env.")
    p.add_argument(
        "--dest", required=True, help="Local path to write the checkpoint to."
    )
    p.add_argument("--filename", default="last.pt")
    p.set_defaults(func=_run_fetch_checkpoint)


def _run_fetch_checkpoint(args: argparse.Namespace) -> None:
    repo_id = args.repo_id or os.environ.get("HF_MODEL_REPO", _DEFAULT_CKPT_REPO)
    print(f"[housekeeping] fetch-checkpoint: {repo_id} -> {args.dest}")
    fetch_checkpoint(
        repo_id=repo_id, dest=Path(args.dest), filename=args.filename
    )


# --- publish-checkpoint ---------------------------------------------------- #


def _render_model_card(repo_id: str, step: object, cfg_yaml: str) -> str:
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
    commit_message: str | None = None,
    private: bool = True,
) -> str:
    import torch
    import yaml
    from huggingface_hub import HfApi

    ckpt_path = Path(ckpt_path)
    tok = os.environ["HF_TOKEN"]

    api = HfApi(token=tok)
    api.create_repo(
        repo_id=repo_id, repo_type="model", exist_ok=True, private=private
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    step = ckpt.get("step", "?")
    cfg = ckpt.get("cfg", {})
    cfg_yaml = yaml.safe_dump(cfg, sort_keys=False)

    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )
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

    url = f"https://huggingface.co/{repo_id}"
    print(f"[publish] done: {url}")
    return url


def _add_publish_checkpoint(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", required=True, help="Path to last.pt")
    p.add_argument(
        "--repo-id", default=None, help="Default: $HF_MODEL_REPO env."
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public (default: private).",
    )
    p.add_argument("--commit-message", default=None)
    p.set_defaults(func=_run_publish_checkpoint)


def _run_publish_checkpoint(args: argparse.Namespace) -> None:
    repo_id = args.repo_id or os.environ.get(
        "HF_MODEL_REPO", _DEFAULT_CKPT_REPO
    )
    print(f"[housekeeping] publish-checkpoint: {args.ckpt} -> {repo_id}")
    publish_checkpoint(
        ckpt_path=Path(args.ckpt),
        repo_id=repo_id,
        commit_message=args.commit_message,
        private=not args.public,
    )


# --- dispatch -------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(prog="housekeeping.py")
    sub = ap.add_subparsers(dest="command", required=True)

    _add_download(sub.add_parser("download", help="Download raw archives."))
    _add_make_manifests(
        sub.add_parser(
            "make-manifests",
            help="Write train/val manifests over attached raw datasets (no transcode).",
        )
    )
    _add_publish_checkpoint(
        sub.add_parser("publish-checkpoint", help="Upload a checkpoint to HF Hub.")
    )
    _add_fetch_checkpoint(
        sub.add_parser("fetch-checkpoint", help="Download a checkpoint from HF Hub.")
    )

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
