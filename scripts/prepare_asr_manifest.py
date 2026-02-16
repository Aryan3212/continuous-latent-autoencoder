import argparse
import json
import csv
import pathlib
import pandas as pd
from typing import List

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=10000, help="Limit ASR subset size for speed")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)
    # id, sentence, split
    
    data = []
    root = pathlib.Path(args.audio_root)
    
    print(f"Loaded {len(df)} rows from {args.csv_path}")
    
    # Filter only files that exist in manifest_dir's split files?
    # Actually, we can just check if file exists.
    # But user wants probing on training data (subset).
    
    count = 0
    for _, row in df.iterrows():
        fname = f"{row['id']}.mp3"
        path = root / fname
        if path.exists():
            data.append({
                "audio_filepath": str(path.absolute()),
                "text": row["sentence"],
                "duration": 5.0 # Placeholder, eval_asr re-measures or handles it
            })
            count += 1
            if count >= args.limit:
                break
    
    print(f"Found {len(data)} valid files (limit={args.limit})")
    
    # Split
    split_idx = int(len(data) * args.train_frac)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    out_dir = pathlib.Path(args.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "train_asr.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(out_dir / "val_asr.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Wrote {len(train_data)} train and {len(val_data)} val samples to {out_dir}")

if __name__ == "__main__":
    main()
