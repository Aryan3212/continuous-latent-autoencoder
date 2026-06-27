#!/usr/bin/env python
"""Correctness gate for SIGReg under DDP.

SIGReg is a batch-statistics loss (it averages the empirical characteristic
function over the N batch points and scales by N), so naive sharding would
change its gradient. train.py handles this by all-gathering SIGReg's input
across ranks (`_gather_with_grad`) and scaling the term by `world_size` to
cancel DDP's 1/W gradient averaging. This test asserts that recipe reproduces
the single-GPU full-batch gradient EXACTLY, using 2 gloo/CPU processes so it
runs anywhere (no 2 GPUs needed).

    uv run python scripts/ddp_sigreg_equivalence_test.py

Prints PASS/FAIL and the max abs gradient difference.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.sigreg import SIGReg  # noqa: E402
from schema import SIGRegCfg  # noqa: E402
from train import _gather_with_grad  # the exact gather train.py uses  # noqa: E402

N, D, STEP, SIG_W = 8, 16, 7, 0.05  # N must be divisible by world_size


def _make(seed: int = 0):
    """Deterministic (input x, projector lin, sigreg) shared by both paths."""
    torch.manual_seed(seed)
    x = torch.randn(N, D)
    lin = nn.Linear(D, D)
    sigreg = SIGReg(D, SIGRegCfg(num_slices=64))
    return x, lin, sigreg


def reference_grad() -> torch.Tensor:
    """Single-process, full-batch: the ground-truth gradient (world_size==1)."""
    x, lin, sigreg = _make()
    loss = SIG_W * 1.0 * sigreg(lin(x), step=STEP)  # sig_scale == 1 single-GPU
    loss.backward()
    return lin.weight.grad.clone()


def _worker(rank: int, world_size: int, ref_grad: torch.Tensor, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    x, lin, sigreg = _make()  # identical init; DDP also broadcasts rank 0's weights
    ddp = DDP(lin)

    half = N // world_size
    x_local = x[rank * half : (rank + 1) * half]
    p_local = ddp(x_local)
    p_global = _gather_with_grad(p_local, world_size, rank)  # (N, D) on every rank

    # ×world_size cancels DDP's 1/W gradient averaging -> single-GPU magnitude.
    loss = SIG_W * float(world_size) * sigreg(p_global, step=STEP)
    loss.backward()

    if rank == 0:
        g = ddp.module.weight.grad
        max_diff = (g - ref_grad).abs().max().item()
        ok = torch.allclose(g, ref_grad, atol=1e-5, rtol=1e-3)
        print(f"[ddp-sigreg] world_size={world_size}  max|Δgrad|={max_diff:.3e}  "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(1)
    dist.destroy_process_group()


def main() -> None:
    torch.set_num_threads(1)  # keep gloo/CPU determinism tidy
    ref = reference_grad()
    world_size = 2
    port = 29555
    torch.multiprocessing.spawn(
        _worker, args=(world_size, ref, port), nprocs=world_size, join=True
    )
    print("[ddp-sigreg] gather + ×world_size matches single-GPU full-batch gradient.")


if __name__ == "__main__":
    main()
