#!/usr/bin/env python3
"""Build a reproducible blinded listening bundle from fixed-source audio pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_index(label: str, path: Path) -> tuple[Path, list[dict[str, Any]]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} audio index {path}: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} audio index is empty: {path}")
    for row in rows:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str)
            for key in ("item_id", "original", "reconstruction")
        ):
            raise ValueError(f"{label} has a malformed audio-index row")
    return path.parent, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_items", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--condition",
        action="append",
        nargs=2,
        metavar=("LABEL", "AUDIO_INDEX_JSON"),
        required=True,
    )
    args = parser.parse_args()
    if args.num_items < 1:
        parser.error("--num_items must be positive")
    labels = [label for label, _ in args.condition]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        parser.error("provide at least two uniquely labelled conditions")

    indexed = {
        label: _load_index(label, Path(path)) for label, path in args.condition
    }
    if args.out_dir.exists():
        raise FileExistsError(
            f"refusing to replace existing listening-study directory: {args.out_dir}"
        )
    reference_label = labels[0]
    reference_root, reference_rows = indexed[reference_label]
    reference_by_id = {row["item_id"]: row for row in reference_rows}
    common_ids = [
        row["item_id"]
        for row in reference_rows
        if all(row["item_id"] in {candidate["item_id"] for candidate in rows}
               for _, rows in indexed.values())
    ]
    if len(common_ids) < args.num_items:
        raise ValueError(
            f"only {len(common_ids)} fixed-source items are shared by every condition; "
            f"requested {args.num_items}"
        )

    rng = random.Random(args.seed)
    selected_ids = common_ids[:]
    rng.shuffle(selected_ids)
    selected_ids = selected_ids[: args.num_items]
    shuffled_labels = labels[:]
    rng.shuffle(shuffled_labels)
    blinded_codes = {
        label: f"C{index + 1:02d}" for index, label in enumerate(shuffled_labels)
    }

    public_dir = args.out_dir / "public"
    audio_dir = public_dir / "audio"
    private_dir = args.out_dir / "private"
    audio_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)

    maps = {
        label: {row["item_id"]: row for row in rows}
        for label, (_, rows) in indexed.items()
    }
    trial_rows: list[dict[str, str]] = []
    hidden_items: list[dict[str, Any]] = []
    public_hashes: list[dict[str, Any]] = []
    for public_index, item_id in enumerate(selected_ids, start=1):
        item_name = f"item_{public_index:03d}"
        item_dir = audio_dir / item_name
        item_dir.mkdir()
        reference_row = reference_by_id[item_id]
        reference_path = reference_root / reference_row["original"]
        reference_hash = _sha256(reference_path)
        shutil.copy2(reference_path, item_dir / "reference.wav")
        public_item_hashes: dict[str, Any] = {
            "item": item_name,
            "reference": {
                "path": f"audio/{item_name}/reference.wav",
                "sha256": _sha256(item_dir / "reference.wav"),
            },
            "options": [],
        }
        hidden_condition_hashes: dict[str, Any] = {}

        for label, (root, _) in indexed.items():
            candidate_original = root / maps[label][item_id]["original"]
            if _sha256(candidate_original) != reference_hash:
                raise ValueError(
                    f"original waveform mismatch for {item_id}: {reference_label} vs {label}"
                )
            code = blinded_codes[label]
            shutil.copy2(
                root / maps[label][item_id]["reconstruction"], item_dir / f"{code}.wav"
            )
            candidate_hash = _sha256(item_dir / f"{code}.wav")
            public_item_hashes["options"].append(
                {
                    "code": code,
                    "path": f"audio/{item_name}/{code}.wav",
                    "sha256": candidate_hash,
                }
            )
            hidden_condition_hashes[label] = {
                "code": code,
                "sha256": candidate_hash,
            }

        presentation = list(blinded_codes.values())
        rng.shuffle(presentation)
        trial: dict[str, str] = {
            "item": item_name,
            "reference": f"audio/{item_name}/reference.wav",
        }
        for option_index, code in enumerate(presentation, start=1):
            trial[f"option_{option_index}"] = f"audio/{item_name}/{code}.wav"
        trial_rows.append(trial)
        hidden_items.append(
            {
                "public_item": item_name,
                "source_item_id": item_id,
                "source_audio_filepath": reference_row.get("audio_filepath"),
                "presentation_order": presentation,
                "reference_sha256": reference_hash,
                "conditions": hidden_condition_hashes,
            }
        )
        public_hashes.append(public_item_hashes)

    fieldnames = ["item", "reference"] + [
        f"option_{index}" for index in range(1, len(labels) + 1)
    ]
    with (public_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)
    rating_fields = ["participant_id", "item", "option", "naturalness_1_to_5", "similarity_1_to_5", "notes"]
    with (public_dir / "ratings_template.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(rating_fields)
    (public_dir / "stimulus_manifest.json").write_text(
        json.dumps(
            {
                "hash_algorithm": "sha256",
                "seed": args.seed,
                "items": public_hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (private_dir / "condition_key.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "condition_codes": blinded_codes,
                "items": hidden_items,
                "source_indexes": {
                    label: str(Path(path).resolve()) for label, path in args.condition
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (public_dir / "README.md").write_text(
        "# Blinded reconstruction listening bundle\n\n"
        "This package prepares stimuli only; it does not collect human ratings. "
        "For each item, listen to the reference and every randomized option. "
        "Record naturalness (1 = very unnatural, 5 = completely natural) and "
        "similarity to the reference (1 = very different, 5 = nearly identical) "
        "in `ratings_template.csv`. Keep `../private/condition_key.json` hidden "
        "until ratings are frozen. Report participant count, exclusions, rating "
        "instructions, and uncertainty estimates in the paper.\n",
        encoding="utf-8",
    )
    print(f"Wrote blinded stimuli to {public_dir}")
    print(f"Keep the condition key private: {private_dir / 'condition_key.json'}")


if __name__ == "__main__":
    main()
