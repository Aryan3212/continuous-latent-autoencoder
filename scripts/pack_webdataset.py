import os
import json
import argparse
import io
import torch
import torchaudio
import webdataset as wds
from tqdm import tqdm
from pathlib import Path

def process_audio(audio_path, target_sr=16000):
    try:
        # Check if it's already a bytes-like object or a path
        if isinstance(audio_path, (str, Path)):
            waveform, sample_rate = torchaudio.load(audio_path)
        else:
            # Assume it's a file-like object or bytes
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_path))
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
            
        return waveform, target_sr
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None

def precompute_stfts(waveform, fft_sizes=[256, 512, 1024, 2048], hop_ratio=0.25, win_ratio=1.0):
    mags = {}
    for n_fft in fft_sizes:
        hop = max(1, int(n_fft * hop_ratio))
        win_len = max(1, int(n_fft * win_ratio))
        window = torch.hann_window(win_len)
        
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_len,
            window=window,
            center=True,
            return_complex=True,
        )
        mags[str(n_fft)] = stft.abs().half() # Use half precision to save space
    return mags

def pack_from_jsonl(args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pattern = str(output_path / f"{args.name}-%06d.tar")
    
    count = 0
    with wds.ShardWriter(pattern, maxsize=args.max_size, maxcount=args.max_count) as sink:
        with open(args.input_manifest, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Packing {args.name}"):
                if args.limit and count >= args.limit:
                    break
                
                data = json.loads(line)
                audio_path = data['audio_filepath']
                
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(args.base_dir, audio_path)

                waveform, sr = process_audio(audio_path, target_sr=args.target_sr)
                if waveform is None:
                    continue
                
                # Convert waveform to FLAC bytes
                buffer = io.BytesIO()
                torchaudio.save(buffer, waveform, sr, format="flac")
                audio_bytes = buffer.getvalue()
                
                key = data.get('id', Path(audio_path).stem)
                
                sample = {
                    "__key__": key,
                    "flac": audio_bytes,
                }
                
                if args.precompute_stft:
                    mags = precompute_stfts(waveform)
                    stft_buffer = io.BytesIO()
                    torch.save(mags, stft_buffer)
                    sample["stft.pth"] = stft_buffer.getvalue()

                sample["json"] = json.dumps({
                    "text": data.get('text', ""),
                    "duration": waveform.shape[1] / sr,
                    "speaker_id": data.get('speaker_id', ""),
                    "id": key,
                    "dataset": data.get('dataset', args.name)
                }).encode('utf-8')
                
                sink.write(sample)
                count += 1

def main():
    parser = argparse.ArgumentParser(description="Pack audio dataset into WebDataset shards.")
    parser.add_argument("--input_manifest", type=str, required=True, help="Path to input JSONL manifest")
    parser.add_argument("--manifest_type", type=str, choices=['jsonl', 'parquet'], default='jsonl')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store shards")
    parser.add_argument("--name", type=str, default="dataset", help="Prefix for shard names")
    parser.add_argument("--max_size", type=float, default=1e9, help="Max size of each shard in bytes (default 1GB)")
    parser.add_argument("--max_count", type=int, default=10000, help="Max number of items per shard")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory for relative paths in manifest")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to pack")
    parser.add_argument("--precompute_stft", action="store_true", help="Precompute STFT magnitudes for loss calculation")
    
    args = parser.parse_args()
    
    if args.manifest_type == 'jsonl':
        pack_from_jsonl(args)
    else:
        # Note: pack_from_parquet is not updated here for brevity but should follow same pattern
        print("Parquet packing with STFT precomputation not yet implemented in this example.")
        # ... (implementation would be similar to pack_from_jsonl)

if __name__ == "__main__":
    main()
