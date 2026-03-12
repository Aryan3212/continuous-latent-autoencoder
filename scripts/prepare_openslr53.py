import argparse
import json
import pathlib
import os

def create_manifest(tsv_path: str, data_root: str, output_path: str):
    """
    Creates a JSONL manifest from OpenSLR53 utt_spk_text.tsv.
    """
    data_root_path = pathlib.Path(data_root)
    manifest_data = []

    print(f"Processing {tsv_path}...")
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                utt_id = parts[0]
                spk_id = parts[1]
                text = parts[2]
                
                # Audio files are in subfolders named after the first two characters of the ID
                subfolder = utt_id[:2]
                audio_path = data_root_path / subfolder / f"{utt_id}.flac"
                
                if not audio_path.exists():
                    continue
                
                entry = {
                    "audio_filepath": str(audio_path),
                    "text": text,
                    "duration": None,
                    "id": utt_id,
                    "speaker_id": spk_id
                }
                manifest_data.append(entry)

    print(f"Found {len(manifest_data)} valid audio files.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in manifest_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Manifest saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", default="data/Bengali_Speech_Data/OpenSLR53/asr_bengali/utt_spk_text.tsv")
    parser.add_argument("--data_root", default="data/Bengali_Speech_Data/OpenSLR53/asr_bengali/data")
    parser.add_argument("--output_path", default="openslr53_full.jsonl")
    args = parser.parse_args()
    
    create_manifest(args.tsv_path, args.data_root, args.output_path)
