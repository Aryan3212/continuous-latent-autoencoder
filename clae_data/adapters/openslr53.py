from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

from clae_data.adapters.base import DatasetAdapter
from clae_data.schema import Record

# OpenSLR-53 is sharded into 10 numeric parts plus 5 alphabetic parts.
_PARTS: tuple = tuple(list(range(10)) + ["a", "b", "c", "d", "e"])
_BASE_URL = "https://www.openslr.org/resources/53/asr_bengali_{part}.zip"


class OpenSLR53Adapter(DatasetAdapter):
    name = "openslr53"
    language = "bn"
    requires_credentials = ()

    def download(self, dest_root: Path) -> Path:
        out_dir = dest_root / "OpenSLR53"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Marker subdir written by the upstream zips; treat its existence as
        # "already extracted" for whatever subset of parts has been pulled.
        for part in _PARTS:
            zip_path = out_dir / f"asr_bengali_{part}.zip"
            # Heuristic: skip a part if any of its expected payload exists.
            # The zip lays files into ``asr_bengali/data/<2char>/<utt>.flac``;
            # if the tsv is present we assume we got at least that part.
            tsv = out_dir / "asr_bengali" / "utt_spk_text.tsv"
            if tsv.exists() and not zip_path.exists():
                # We can't tell per-part without parsing the tsv; just skip
                # re-downloading individual parts if the tsv is in place.
                continue
            if zip_path.exists():
                # Stale zip from a previous failed run — re-extract then drop.
                pass
            else:
                url = _BASE_URL.format(part=part)
                print(f"[openslr53] downloading part {part} from {url}")
                try:
                    urllib.request.urlretrieve(url, zip_path)
                except Exception as e:
                    print(f"[openslr53] failed to download part {part}: {e}")
                    continue

            print(f"[openslr53] extracting part {part}")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(out_dir)
                zip_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[openslr53] failed to extract part {part}: {e}")

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
