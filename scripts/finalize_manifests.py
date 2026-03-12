import json
import random
import os

random.seed(42)

def load_manifest(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_manifest(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    # 1. IndicVoices: split test into val and test (50/50)
    print("Splitting IndicVoices test into val and test...")
    indic_test = load_manifest("data/manifests/indicvoices_test.jsonl")
    if indic_test:
        random.shuffle(indic_test)
        mid = len(indic_test) // 2
        save_manifest(indic_test[:mid], "data/manifests/indicvoices_val.jsonl")
        save_manifest(indic_test[mid:], "data/manifests/indicvoices_test_split.jsonl")

    # 2. OOD_Speech: create 20% splits for ASR eval
    print("\nCreating 20% splits for OOD Speech ASR eval...")
    ood_train = load_manifest("data/manifests/ood_speech_train.jsonl")
    if ood_train:
        asr_train_size = int(len(ood_train) * 0.20)
        save_manifest(ood_train[:asr_train_size], "data/manifests/asr_eval_train.jsonl")
        print(f"Saved {asr_train_size} samples to asr_eval_train.jsonl")

    ood_val = load_manifest("data/manifests/ood_speech_val.jsonl")
    if ood_val:
        asr_val_size = int(len(ood_val) * 0.20)
        save_manifest(ood_val[:asr_val_size], "data/manifests/asr_eval_val.jsonl")
        print(f"Saved {asr_val_size} samples to asr_eval_val.jsonl")

    # 3. Combine Pretraining Train Manifests
    print("\nCombining pretraining train manifests...")
    train_files = [
        "data/manifests/openslr53_train.jsonl",
        "data/manifests/regspeech12_train.jsonl",
        "data/manifests/indicvoices_train.jsonl",
        "data/manifests/subak_ko_train.jsonl",
        "data/manifests/ood_speech_train.jsonl"
    ]
    combined_train = []
    for f in train_files:
        data = load_manifest(f)
        print(f"Loaded {len(data)} from {f}")
        combined_train.extend(data)

    # Shuffle combined train manifest to ensure batches have diverse datasets
    random.shuffle(combined_train)
    save_manifest(combined_train, "data/manifests/combined_train.jsonl")
    print(f"Saved {len(combined_train)} total samples to combined_train.jsonl")

    # 4. Combine Pretraining Val Manifests
    print("\nCombining pretraining val manifests...")
    val_files = [
        "data/manifests/openslr53_val.jsonl",
        "data/manifests/regspeech12_val.jsonl",
        "data/manifests/indicvoices_val.jsonl",
        "data/manifests/subak_ko_val.jsonl",
        "data/manifests/ood_speech_val.jsonl"
    ]
    combined_val = []
    for f in val_files:
        data = load_manifest(f)
        print(f"Loaded {len(data)} from {f}")
        combined_val.extend(data)

    save_manifest(combined_val, "data/manifests/combined_val.jsonl")
    print(f"Saved {len(combined_val)} total samples to combined_val.jsonl")

if __name__ == "__main__":
    main()
