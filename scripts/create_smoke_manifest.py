import json
import os
import glob

# Get first 20 mp3 files
files = glob.glob("data/bengaliai_speech/train_mp3s/*.mp3")
files = sorted(files)[:20]

with open("data/manifest_train.jsonl", "w") as f:
    for path in files:
        # Use absolute path
        abs_path = os.path.abspath(path)
        item = {"audio_filepath": abs_path}
        f.write(json.dumps(item) + "\n")
