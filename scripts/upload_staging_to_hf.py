#!/usr/bin/env python
"""Upload the processed `staging/` corpus to a (private) HF dataset repo as plain
tar shards, structure-preserving so it round-trips with a simple extract:

    # on the compute node, after downloading the .tar files into <dst>/:
    for t in *.tar; do tar -xf "$t" -C staging; done
    # -> recreates staging/audio/<dataset>/*.flac and staging/manifests/*.jsonl

Why shards (not one big tar, not WebDataset): this box has only ~40 GB free, so we
build ONE ~25 GB tar at a time, upload it, delete it, then build the next. Nothing
fancy in the format — `tar -xf` gives back the identical tree, so data_loading.py
needs no changes.

Resumable: shard membership is deterministic (sorted file list, greedy fill), and
shards already present in the repo are skipped. Safe to re-run after a disconnect.

Usage:
    uv run python scripts/upload_staging_to_hf.py --repo <user>/<name> [--dry-run]
    uv run python scripts/upload_staging_to_hf.py --repo <user>/<name> --shard-gb 25

Token: reads HF_TOKEN from the environment (a WRITE token). The repo is created
private by default.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

GB = 1024 ** 3


def plan_shards(staging: pathlib.Path, shard_bytes: int):
    """Deterministic shard plan. Returns list of (shard_name, [rel_paths], total_bytes).

    One stream of shards per dataset dir (so a shard never straddles datasets, which
    keeps names stable if you later add a dataset). Files sorted for determinism.
    """
    shards = []
    audio = staging / "audio"
    for ds_dir in sorted(p for p in audio.iterdir() if p.is_dir()):
        files = sorted(ds_dir.rglob("*"))
        files = [f for f in files if f.is_file()]
        idx, cur, cur_bytes = 0, [], 0
        for f in files:
            sz = f.stat().st_size
            if cur and cur_bytes + sz > shard_bytes:
                shards.append((f"audio_{ds_dir.name}_{idx:03d}.tar",
                               [str(p.relative_to(staging)) for p in cur], cur_bytes))
                idx += 1
                cur, cur_bytes = [], 0
            cur.append(f)
            cur_bytes += sz
        if cur:
            shards.append((f"audio_{ds_dir.name}_{idx:03d}.tar",
                           [str(p.relative_to(staging)) for p in cur], cur_bytes))
    # manifests as one shard (small)
    man = staging / "manifests"
    if man.is_dir():
        man_files = sorted(str(p.relative_to(staging)) for p in man.rglob("*") if p.is_file())
        man_bytes = sum((staging / m).stat().st_size for m in man_files)
        shards.append(("manifests.tar", man_files, man_bytes))
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. user/bengali-speech-16k")
    ap.add_argument("--staging", default="staging")
    ap.add_argument("--shard-gb", type=float, default=25.0)
    ap.add_argument("--tmp", default="_hf_upload_tmp",
                    help="scratch dir for the current tar; MUST be on disk, not tmpfs/RAM")
    ap.add_argument("--public", action="store_true", help="override the private default")
    ap.add_argument("--dry-run", action="store_true", help="print the shard plan and exit")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # fast large uploads if installed
    staging = pathlib.Path(args.staging).resolve()
    if not (staging / "audio").is_dir():
        sys.exit(f"no audio/ under {staging}")

    shard_bytes = int(args.shard_gb * GB)
    shards = plan_shards(staging, shard_bytes)
    total = sum(b for _, _, b in shards)
    print(f"plan: {len(shards)} shards, {total/GB:.1f} GB total")
    for name, files, b in shards:
        print(f"  {name:34s} {b/GB:6.2f} GB  {len(files):>8d} files")
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set (export a WRITE token)")
    api = HfApi(token=token)
    print("auth:", api.whoami()["name"])
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    existing = set(api.list_repo_files(args.repo, repo_type="dataset"))

    tmp = pathlib.Path(args.tmp).resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    # guard: refuse a tmp dir that lives on tmpfs (would eat RAM and risk an OOM)
    fstype = subprocess.run(["stat", "-f", "-c", "%T", str(tmp)],
                            capture_output=True, text=True).stdout.strip()
    if fstype == "tmpfs":
        sys.exit(f"--tmp {tmp} is tmpfs (RAM-backed); point it at real disk")

    for name, files, b in shards:
        if name in existing:
            print(f"skip (already uploaded): {name}")
            continue
        tar_path = tmp / name
        listfile = tmp / (name + ".list")
        listfile.write_text("\n".join(files) + "\n")
        print(f"taring {name} ({b/GB:.2f} GB, {len(files)} files) ...", flush=True)
        subprocess.run(["tar", "-C", str(staging), "-cf", str(tar_path), "-T", str(listfile)],
                       check=True)
        print(f"uploading {name} ...", flush=True)
        api.upload_file(path_or_fileobj=str(tar_path), path_in_repo=name,
                        repo_id=args.repo, repo_type="dataset",
                        commit_message=f"add {name}")
        tar_path.unlink(missing_ok=True)
        listfile.unlink(missing_ok=True)
        print(f"done: {name}")

    print("all shards uploaded.")


if __name__ == "__main__":
    main()
