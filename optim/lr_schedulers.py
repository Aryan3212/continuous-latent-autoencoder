from __future__ import annotations

import logging
from typing import List, Optional, Union

from torch.optim import Optimizer


class LRScheduler:
    def __init__(self, optimizer: Optimizer, verbose: bool = False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.verbose = verbose
        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])
        self.base_lrs = [group["base_lr"] for group in optimizer.param_groups]
        self.epoch = 0
        self.batch = 0
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def state_dict(self):
        return {"epoch": self.epoch, "batch": self.batch}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.get("epoch", 0)
        self.batch = state_dict.get("batch", 0)
        self._set_lrs()

    def get_last_lr(self) -> List[float]:
        return self._last_lr

    def get_lr(self):
        raise NotImplementedError

    def step_batch(self, batch: Optional[int] = None) -> None:
        if batch is not None:
            self.batch = batch
        else:
            self.batch = self.batch + 1
        self._set_lrs()

    def step_epoch(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch = self.epoch + 1
        self._set_lrs()

    def _set_lrs(self):
        values = self.get_lr()
        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, values)):
            param_group["lr"] = lr
            if self.verbose:
                logging.warning(f"Epoch={self.epoch}, batch={self.batch}: lr group {i} -> {lr:.4e}")
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class Eden(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        lr_epochs: Union[int, float],
        warmup_batches: Union[int, float] = 500.0,
        warmup_start: float = 0.5,
        verbose: bool = False,
    ):
        super().__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.lr_epochs = lr_epochs
        self.warmup_batches = warmup_batches
        if not 0.0 <= warmup_start <= 1.0:
            raise ValueError("warmup_start must be in [0,1]")
        self.warmup_start = warmup_start

    def get_lr(self):
        factor = (
            (self.batch**2 + self.lr_batches**2) / self.lr_batches**2
        ) ** -0.25 * (
            ((self.epoch**2 + self.lr_epochs**2) / self.lr_epochs**2) ** -0.25
        )
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
        )
        return [x * factor * warmup_factor for x in self.base_lrs]


class Eden2(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        warmup_batches: Union[int, float] = 500.0,
        warmup_start: float = 0.5,
        verbose: bool = False,
    ):
        super().__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.warmup_batches = warmup_batches
        if not 0.0 <= warmup_start <= 1.0:
            raise ValueError("warmup_start must be in [0,1]")
        self.warmup_start = warmup_start

    def get_lr(self):
        factor = ((self.batch**2 + self.lr_batches**2) / self.lr_batches**2) ** -0.5
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
        )
        return [x * factor * warmup_factor for x in self.base_lrs]
