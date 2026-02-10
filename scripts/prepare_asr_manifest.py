import json
import csv
import pathlib

# Load the 10% manifest
with open("data/manifest_train_10pct.jsonl", "r") as f:
    manifest = [json.loads(line) for line in f]

# Load train.csv
id_to_sentence = {}
with open("data/bengaliai_speech/train.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        id_to_sentence[row["id"]] = row["sentence"]

# Enrich manifest with sentence
enriched = []
for item in manifest:
    # item["audio_filepath"] is like "/.../train_mp3s/b21535ad03b6.mp3"
    path = pathlib.Path(item["audio_filepath"])
    audio_id = path.stem
    if audio_id in id_to_sentence:
        item["sentence"] = id_to_sentence[audio_id]
        enriched.append(item)

print(f"Enriched {len(enriched)} samples out of {len(manifest)}")

# Split into train and val (90/10)
split = int(0.9 * len(enriched))
train_asr = enriched[:split]
val_asr = enriched[split:]

with open("data/manifest_train_10pct_asr.jsonl", "w") as f:
    for item in train_asr:
        f.write(json.dumps(item) + "\n")

with open("data/manifest_val_10pct_asr.jsonl", "w") as f:
    for item in val_asr:
        f.write(json.dumps(item) + "\n")
