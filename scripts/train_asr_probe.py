from __future__ import annotations

import argparse
import json
import pathlib
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import WebDatasetConfig, get_audio_wds, collate_fixed
from eval.common import load_frozen_encoder
from jiwer import wer
from typing import List, Dict, Tuple, Any
from utils.logging import JsonlLogger, maybe_init_wandb

def build_charset(texts: List[str]) -> List[str]:
    chars = set()
    for t in texts:
        chars.update(list(t))
    chars.discard("\n")
    return ["<blank>"] + sorted(list(chars))

def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[c] for c in text if c in vocab]

def greedy_decode(log_probs: torch.Tensor, id2ch: List[str]) -> List[str]:
    pred = log_probs.argmax(dim=-1)  # (B,T)
    outs = []
    for seq in pred.tolist():
        last = None
        chars = []
        for i in seq:
            if i == 0:
                last = i
                continue
            if last != i:
                chars.append(id2ch[i])
            last = i
        outs.append("".join(chars))
    return outs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to main model config")
    parser.add_argument("--ckpt", required=True, help="Path to main model checkpoint")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--out_dir", default="runs/asr_probes")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--wandb_project", default="continuous-latent-ae-probes")
    args = parser.parse_args()

    # Setup Output
    run_id = args.run_id or f"asr_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root = pathlib.Path(args.out_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    
    jsonl = JsonlLogger(str(out_root / "train.jsonl"))
    
    # Load Model
    lm = load_frozen_encoder(args.config, args.ckpt, [])
    device = lm.device
    
    # Init WandB
    wb_cfg = {
        "run": {"wandb": {"enabled": True, "project": args.wandb_project, "name": run_id}},
        "probe": vars(args),
        "encoder_ckpt": args.ckpt
    }
    wb = maybe_init_wandb(wb_cfg, run_id, str(out_root))

    # Dataset
    d_cfg = lm.cfg["data"]
    ds_train = get_audio_wds(WebDatasetConfig(
        urls=args.train_manifest,
        sample_rate=d_cfg["sample_rate"],
        segment_seconds=d_cfg["segment_seconds"],
    ))
    ds_train = ds_train.batched(args.batch_size, collation_fn=collate_fixed)
    dl_train = DataLoader(ds_train, batch_size=None, num_workers=4, pin_memory=True)

    ds_val = get_audio_wds(WebDatasetConfig(
        urls=args.val_manifest,
        sample_rate=d_cfg["sample_rate"],
        segment_seconds=d_cfg["segment_seconds"],
        resampled=False,
        shuffle_size=0,
    ))
    ds_val = ds_val.batched(args.batch_size, collation_fn=collate_fixed)
    dl_val = DataLoader(ds_val, batch_size=None, num_workers=4)

    # Charset / Vocab
    texts = []
    print("Building vocabulary from the first 20,000 samples of the train stream...")
    train_iter = iter(dl_train)
    for batch in train_iter:
        for m in batch["meta"]:
            texts.append(m["sentence"])
        if len(texts) > 20000:
            break
    
    charset = build_charset(texts)
    vocab = {c: i for i, c in enumerate(charset)}
    id2ch = charset
    print(f"Vocab size: {len(charset)}")

    # ASR Head
    d_model = lm.cfg["model"]["encoder"]["d_model"]
    head = nn.Linear(d_model, len(charset)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    best_wer = float("inf")
    step = 0
    train_iter = iter(dl_train)

    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train)
            batch = next(train_iter)

        wav = batch["wav"].to(device, non_blocking=True)
        with torch.no_grad():
            h0 = lm.frontend(wav)
            hE = lm.encoder(h0) 
            feats = hE.transpose(1, 2) 

        logits = head(feats)
        log_probs = logits.log_softmax(dim=-1)

        target_list = []
        target_lens = []
        sentences = [m["sentence"] for m in batch["meta"]]
        for s in sentences:
            encoded = encode_text(s, vocab)
            target_list.append(torch.tensor(encoded, dtype=torch.long))
            target_lens.append(len(encoded))
        
        y_true = torch.cat(target_list).to(device)
        y_lens = torch.tensor(target_lens, dtype=torch.long).to(device)
        input_lens = torch.full((feats.size(0),), feats.size(1), dtype=torch.long).to(device)

        loss = ctc_loss(log_probs.transpose(0, 1), y_true, input_lens, y_lens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            hyp = greedy_decode(log_probs[0:1], id2ch)[0]
            ref = sentences[0]
            audio_id = pathlib.Path(batch["meta"][0]["audio_filepath"]).stem
            stats = {"step": step, "loss": loss.item()}
            jsonl.log(stats)
            if wb: wb.log(stats, step=step)
            print(f"Step {step} | ID: {audio_id}")
            print(f"  Ref: {ref}")
            print(f"  Hyp: {hyp}")
            print(f"  Loss: {loss.item():.4f}")
            print("-" * 30)

        if step > 0 and step % args.val_interval == 0:
            head.eval()
            total_wer = 0
            n_val = 0
            with torch.no_grad():
                for vbatch in dl_val:
                    vwav = vbatch["wav"].to(device)
                    vhE = lm.encoder(lm.frontend(vwav))
                    vlogits = head(vhE.transpose(1, 2))
                    vhyp = greedy_decode(vlogits.log_softmax(dim=-1), id2ch)
                    vref = [m["sentence"] for m in vbatch["meta"]]
                    total_wer += wer(vref, vhyp)
                    n_val += 1
                    if n_val >= 20: break 
            
            avg_wer = total_wer / n_val
            print(f">>> Step {step}, Val WER: {avg_wer:.4f}")
            if wb: wb.log({"val_wer": avg_wer}, step=step)
            
            if avg_wer < best_wer:
                best_wer = avg_wer
                torch.save({
                    "step": step,
                    "head": head.state_dict(),
                    "vocab": vocab,
                    "wer": best_wer
                }, str(out_root / "best_head.pt"))
            head.train()

        step += 1

    # Save final
    torch.save({"step": step, "head": head.state_dict(), "vocab": vocab}, str(out_root / "last_head.pt"))

if __name__ == "__main__":
    main()
