import argparse
import json
import pathlib
import pandas as pd
import os

def create_manifest(excel_path: str, audio_root: str, output_path: str):
    """
    Creates a JSONL manifest from a RegSpeech12 Excel file.
    """
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading {excel_path}: {e}")
        print("Please ensure 'pandas' and 'openpyxl' are installed.")
        return

    audio_root_path = pathlib.Path(audio_root)
    manifest_data = []

    print(f"Processing {len(df)} entries from {excel_path}...")
    
    # Try to find the right columns. Typical names for audio file and transcript:
    id_cols = ['id', 'file_name', 'filename', 'audio', 'path', df.columns[0]]
    text_cols = ['sentence', 'text', 'transcript', 'transcription', df.columns[1]]
    
    id_col = next((col for col in id_cols if col in df.columns), df.columns[0])
    text_col = next((col for col in text_cols if col in df.columns), df.columns[1])
    
    for idx, row in df.iterrows():
        file_id = str(row[id_col])
        text = str(row[text_col])
        
        # Audio path might already have the extension, or not
        audio_path = audio_root_path / file_id
        if not audio_path.exists():
            # Try appending .wav or .mp3 or .flac
            for ext in ['.wav', '.mp3', '.flac']:
                temp_path = audio_root_path / f"{file_id}{ext}"
                if temp_path.exists():
                    audio_path = temp_path
                    break
                    
        if not audio_path.exists():
            continue
            
        entry = {
            "audio_filepath": str(audio_path).replace("\\", "/"),
            "text": text,
            "duration": None,
            "id": file_id
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
    parser.add_argument("excel_path", help="Path to the Excel file (e.g. train.xlsx)")
    parser.add_argument("audio_root", help="Root directory of audio files for this split")
    parser.add_argument("output_path", help="Output JSONL path")
    args = parser.parse_args()
    
    create_manifest(args.excel_path, args.audio_root, args.output_path)
