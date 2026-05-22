"""Shared logic for HF datasets distributed as parquet with inline audio bytes.

Used by IndicVoices, SUBAK.KO, Shrutilipi, Kathbath. The downstream pack
step needs files on disk (torchaudio.load), so we extract inline bytes to
``<raw_dir>/extracted/<dataset>/<id>.<ext>`` the first time iter_records is
called.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence

from clae_data.schema import Record


def hf_snapshot_download(
    repo_id: str,
    dest_dir: Path,
    allow_patterns: Optional[Sequence[str] | str] = None,
) -> Path:
    """Idempotent snapshot_download into ``dest_dir``.

    huggingface_hub already short-circuits on cache hits, so re-running is
    cheap. We only wrap to consistently pass the hardcoded HF_TOKEN.
    """
    from huggingface_hub import snapshot_download
    from clae_data._creds import HF_TOKEN

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hf] snapshot_download {repo_id} -> {dest_dir!s}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest_dir),
        allow_patterns=allow_patterns,
        token=HF_TOKEN,
    )
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
        audio_col = next((c for c in ("audio", "speech") if c in cols), None)
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
