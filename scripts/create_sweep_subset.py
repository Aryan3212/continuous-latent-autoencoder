import json
import random
import os

def create_subset(input_path, output_path, fraction=0.1):
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    random.seed(42)
    subset_size = int(len(data) * fraction)
    subset_data = random.sample(data, subset_size)
    
    print(f"Creating subset: {subset_size} samples (out of {len(data)})")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in subset_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    create_subset("data/manifests/combined_train.jsonl", "data/manifests/sweep_train_subset.jsonl", fraction=0.05)
