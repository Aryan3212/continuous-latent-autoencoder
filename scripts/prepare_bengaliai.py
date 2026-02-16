import pandas as pd
import pathlib
import json
import argparse
from typing import List, Dict, Any

def create_manifest(csv_path: str, audio_root: str, output_path: str):
    """
    Creates a JSONL manifest from a Bengali.AI Speech CSV.
    """
    df = pd.read_csv(csv_path)
    audio_root_path = pathlib.Path(audio_root)
    
    manifest_data = []
    
    # Assuming CSV has columns like 'id', 'sentence', 'path' or similar
    # Adjust based on actual Bengali.AI structure
    # Usually: id, sentence, split, path
    
    print(f"Processing {len(df)} entries from {csv_path}...")
    
    for idx, row in df.iterrows():
        # Adjust column names as needed for your specific CSV
        audio_path = audio_root_path / f"{row['id']}.mp3" # Common format
        if not audio_path.exists():
             # Try other extensions or look for 'path' column
             if 'path' in row:
                 audio_path = audio_root_path / row['path']
             else:
                 continue

        if not audio_path.exists():
            continue

        entry = {
            "audio_filepath": str(audio_path),
            "text": row.get('sentence', ""),
            "duration": None, # Duration often not in CSV, handled by loader or separate pass
            # Add other metadata
            "id": row.get('id', str(idx)),
            "speaker_id": row.get('client_id', None)
        }
        manifest_data.append(entry)

    print(f"Found {len(manifest_data)} valid audio files.")
    
    with open(output_path, 'w') as f:
        for item in manifest_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Manifest saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to train.csv")
    parser.add_argument("audio_root", help="Root directory of audio files")
    parser.add_argument("output_path", help="Output JSONL path")
    args = parser.parse_args()
    
    create_manifest(args.csv_path, args.audio_root, args.output_path)
