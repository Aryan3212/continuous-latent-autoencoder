import argparse
import json
import pathlib
import soundfile as sf
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def audit_file(line_str):
    try:
        item = json.loads(line_str)
        path = item['audio_filepath']
        
        # Check existence
        if not pathlib.Path(path).exists():
            return {'status': 'missing', 'path': path, 'item': item}
            
        # Check validity and duration
        try:
            info = sf.info(path)
            if info.duration < 0.1:
                return {'status': 'too_short', 'duration': info.duration, 'path': path, 'item': item}
            if info.frames == 0:
                return {'status': 'empty', 'path': path, 'item': item}
        except Exception as e:
            return {'status': 'corrupt', 'error': str(e), 'path': path, 'item': item}
            
        return {'status': 'ok', 'duration': info.duration, 'item': item}
    except Exception as e:
        return {'status': 'json_error', 'error': str(e), 'line': line_str}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input manifest to audit")
    parser.add_argument("--output_report", default="audit_report.jsonl", help="Output report")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    print(f"Auditing {args.manifest}...")
    
    with open(args.manifest) as f:
        lines = [line.strip() for line in f if line.strip()]

    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for res in tqdm(executor.map(audit_file, lines), total=len(lines)):
            results.append(res)

    stats = {}
    with open(args.output_report, 'w') as f:
        for res in results:
            status = res['status']
            stats[status] = stats.get(status, 0) + 1
            f.write(json.dumps(res) + "\n")

    print("\nAudit Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Report saved to {args.output_report}")

if __name__ == "__main__":
    main()
