import os
import sys
import pathlib

# Ensure we can import from the scripts directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prepare_regspeech12 import create_manifest as create_regspeech_manifest
from prepare_hf_parquet import create_manifest as create_parquet_manifest

def main():
    manifests_dir = "data/manifests"
    os.makedirs(manifests_dir, exist_ok=True)
    
    # 1. RegSpeech12
    print("=== Processing RegSpeech12 ===")
    regspeech_base = "data/Bengali_Speech_Data/RegSpeech12"
    create_regspeech_manifest(
        os.path.join(regspeech_base, "train.xlsx"),
        os.path.join(regspeech_base, "train"),
        os.path.join(manifests_dir, "regspeech12_train.jsonl")
    )
    create_regspeech_manifest(
        os.path.join(regspeech_base, "valid.xlsx"),
        os.path.join(regspeech_base, "valid"),
        os.path.join(manifests_dir, "regspeech12_val.jsonl")
    )
    create_regspeech_manifest(
        os.path.join(regspeech_base, "test.xlsx"),
        os.path.join(regspeech_base, "test"),
        os.path.join(manifests_dir, "regspeech12_test.jsonl")
    )

    # 2. IndicVoices
    print("\n=== Processing IndicVoices ===")
    indicvoices_base = "data/Bengali_Speech_Data/IndicVoices/Bengali"
    indicvoices_out_dir = "data/Bengali_Speech_Data/IndicVoices/extracted_audio"
    create_parquet_manifest(
        indicvoices_base,
        indicvoices_out_dir,
        os.path.join(manifests_dir, "indicvoices_train.jsonl"),
        "train"
    )
    create_parquet_manifest(
        indicvoices_base,
        indicvoices_out_dir,
        os.path.join(manifests_dir, "indicvoices_test.jsonl"),
        "test"
    )

    # 3. SUBAK_KO
    print("\n=== Processing SUBAK_KO ===")
    subak_ko_base = "data/Bengali_Speech_Data/SUBAK_KO/Data"
    subak_ko_out_dir = "data/Bengali_Speech_Data/SUBAK_KO/extracted_audio"
    create_parquet_manifest(
        subak_ko_base,
        subak_ko_out_dir,
        os.path.join(manifests_dir, "subak_ko_train.jsonl"),
        "train"
    )
    create_parquet_manifest(
        subak_ko_base,
        subak_ko_out_dir,
        os.path.join(manifests_dir, "subak_ko_val.jsonl"),
        "validation"
    )
    create_parquet_manifest(
        subak_ko_base,
        subak_ko_out_dir,
        os.path.join(manifests_dir, "subak_ko_test.jsonl"),
        "test"
    )
    
    print("\nAll remaining datasets have been processed successfully!")

if __name__ == "__main__":
    main()
