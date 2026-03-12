import argparse
import json
import pathlib
import os

def create_manifest(parquet_dir: str, audio_output_dir: str, output_path: str, split_pattern: str):
    """
    Creates a JSONL manifest from Hugging Face Parquet files using PyArrow.
    This avoids the datasets library trying to load torchcodec.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import soundfile as sf
    import io
    
    audio_root_path = pathlib.Path(audio_output_dir)
    audio_root_path.mkdir(parents=True, exist_ok=True)
    manifest_data = []

    print(f"Loading parquet files from {parquet_dir} matching {split_pattern}...")
    
    # Find parquet files
    parquet_files = list(pathlib.Path(parquet_dir).glob(f"{split_pattern}-*.parquet"))
    if not parquet_files:
        parquet_files = list(pathlib.Path(parquet_dir).glob("*.parquet"))
        if not parquet_files:
            print("No parquet files found.")
            return

    count = 0
    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
        except Exception as e:
            print(f"Failed to read {pf}: {e}")
            continue
            
        # Get column names
        cols = table.column_names
        
        # We need an audio column and a text column
        audio_col = next((c for c in ['audio', 'speech'] if c in cols), None)
        if not audio_col:
            continue
            
        text_col = next((c for c in ['text', 'sentence', 'transcript', 'transcription'] if c in cols), None)
        id_col = next((c for c in ['id', 'file', 'path'] if c in cols), None)
        
        for i in range(table.num_rows):
            count += 1
            audio_field = table[audio_col][i].as_py()
            
            # Extract raw bytes or array
            audio_bytes = None
            if isinstance(audio_field, dict):
                if 'bytes' in audio_field and audio_field['bytes'] is not None:
                    audio_bytes = audio_field['bytes']
            elif isinstance(audio_field, bytes):
                audio_bytes = audio_field
                
            file_id = None
            if id_col:
                val = table[id_col][i].as_py()
                if val:
                    file_id = str(pathlib.Path(str(val)).stem)
            
            if not file_id:
                file_id = f"audio_{count:08d}"
                
            audio_path = audio_root_path / f"{file_id}.wav"
            
            if not audio_path.exists() and audio_bytes:
                # Save raw bytes
                with open(audio_path, 'wb') as f:
                    f.write(audio_bytes)
                    
            if not audio_path.exists():
                continue
                
            text = ""
            if text_col:
                text_val = table[text_col][i].as_py()
                if text_val:
                    text = str(text_val)
                    
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
    parser.add_argument("parquet_dir", help="Directory containing parquet files")
    parser.add_argument("audio_output_dir", help="Directory to save extracted audio files")
    parser.add_argument("output_path", help="Output JSONL path")
    parser.add_argument("--split", default="train", help="Split name pattern (train, test, validation)")
    args = parser.parse_args()
    
    create_manifest(args.parquet_dir, args.audio_output_dir, args.output_path, args.split)