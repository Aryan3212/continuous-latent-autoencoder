"""Rewrite audio_filepath in JSONL manifests to absolute paths.

Handles both Windows backslash paths and relative forward-slash paths.
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _absolutize(path: str) -> str:
    fixed = path.replace("\\", "/")
    if not os.path.isabs(fixed):
        fixed = os.path.normpath(os.path.join(REPO_ROOT, fixed))
    return fixed

for mpath in sys.argv[1:]:
    rows = []
    changed = 0
    with open(mpath) as f:
        for line in f:
            row = json.loads(line)
            old = row.get("audio_filepath", "")
            if old and not os.path.isabs(old):
                row["audio_filepath"] = _absolutize(old)
                changed += 1
            rows.append(row)
    with open(mpath, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Fixed {changed}/{len(rows)} rows in {mpath}")
