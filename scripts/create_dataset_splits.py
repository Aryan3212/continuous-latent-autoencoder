import argparse
import json
import random
import pathlib
from typing import List, Dict, Any

def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_manifest(data: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Deterministic dataset splitter")
    parser.add_argument("input_manifest", help="Path to input JSONL manifest")
    parser.add_argument("--val_pct", type=float, default=0.1, help="Validation percentage (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", default="data/manifests", help="Output directory")
    parser.add_argument("--name", default="dataset", help="Base name for output files")
    args = parser.parse_args()

    random.seed(args.seed)
    
    print(f"Loading manifest from {args.input_manifest}...")
    data = load_manifest(args.input_manifest)
    print(f"Total samples: {len(data)}")

    # Shuffle deterministically
    random.shuffle(data)

    val_size = int(len(data) * args.val_pct)
    val_data = data[:val_size]
    train_data = data[val_size:]

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{args.name}_train.jsonl"
    val_path = out_dir / f"{args.name}_val.jsonl"

    print(f"Saving {len(train_data)} training samples to {train_path}")
    save_manifest(train_data, str(train_path))

    print(f"Saving {len(val_data)} validation samples to {val_path}")
    save_manifest(val_data, str(val_path))

    print("Done.")

if __name__ == "__main__":
    main()
