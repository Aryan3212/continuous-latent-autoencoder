import pandas as pd
import json
import pathlib
import random
from tqdm import tqdm

def main():
    # Setup paths
    root_dir = pathlib.Path(__file__).parent.parent
    csv_path = root_dir / "data" / "bengaliai-speech" / "train.csv"
    audio_dir = root_dir / "data" / "bengaliai-speech" / "train_mp3s"
    output_dir = root_dir / "data" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    samples = []
    missing_count = 0

    print("Verifying audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_id = row['id']
        sentence = row['sentence']
        
        # We use relative path from project root for better portability
        audio_rel_path = f"data/bengaliai-speech/train_mp3s/{audio_id}.mp3"
        audio_full_path = root_dir / audio_rel_path
        
        if audio_full_path.exists():
            samples.append({
                "audio_filepath": str(audio_rel_path),
                "text": sentence,
                "id": audio_id
            })
        else:
            missing_count += 1

    print(f"Total samples found in CSV: {len(df)}")
    print(f"Valid audio files found: {len(samples)}")
    print(f"Missing audio files: {missing_count}")

    # Deterministic split
    random.seed(42)
    random.shuffle(samples)

    # 90% train, 5% val, 5% test
    n = len(samples)
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    def save_manifest(data, name):
        path = output_dir / f"experiment_v2_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} samples to {path}")

    save_manifest(train_samples, "train")
    save_manifest(val_samples, "val")
    save_manifest(test_samples, "test")

    # Also create a small subset for quick smoke tests (e.g. 20% of train)
    smoke_samples = train_samples[:int(len(train_samples) * 0.2)]
    save_manifest(smoke_samples, "train_20pct")

if __name__ == "__main__":
    main()
