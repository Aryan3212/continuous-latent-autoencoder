import json
import os
import glob
import random

# Set seed for reproducibility
random.seed(42)

# Find all mp3 files
files = glob.glob("data/bengaliai_speech/train_mp3s/*.mp3")
total_files = len(files)

# Calculate 10%
num_samples = int(total_files * 0.1)
if num_samples == 0 and total_files > 0:
    num_samples = 1  # Ensure at least one file if data exists

# Randomly sample
sampled_files = random.sample(files, num_samples)

print(f"Found {total_files} files. Sampling {num_samples} (10%).")

output_path = "data/manifest_train_10pct.jsonl"
with open(output_path, "w") as f:
    for path in sampled_files:
        abs_path = os.path.abspath(path)
        item = {"audio_filepath": abs_path}
        f.write(json.dumps(item) + "\n")

print(f"Created {output_path}")
