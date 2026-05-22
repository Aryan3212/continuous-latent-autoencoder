"""Inline detached CTC probe for live diagnosis of encoder representation quality.

Trains a small linear head on `hE.detach()` (encoder output) against character
CTC targets pulled from a parallel ASR manifest. Runs alongside SSL training
but never backpropagates into the encoder. Logs probe loss and greedy WER to
W&B at a user-defined interval so latent quality can be tracked without
waiting for the full periodic ASR eval.

Independent of `eval/eval_asr.py` (which extracts features into RAM offline,
trains for many steps, and is invoked from `run_all.py`). The inline probe is
strictly a *monitor*; treat the WER as a relative trend, not an absolute number.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from data.dataset import WebDatasetConfig, collate_fixed, get_audio_wds
from eval.common import BLANK_IDX, build_charset, greedy_decode_ctc


class InlineProbe:
    def __init__(
        self,
        cfg_probe: Dict[str, Any],
        sample_rate: int,
        encoder_dim: int,
        device: torch.device,
        out_root: pathlib.Path,
    ):
        self.enabled = bool(cfg_probe.get("enabled", False))
        if not self.enabled:
            return

        self.device = device
        self.step_interval = max(1, int(cfg_probe.get("step_interval", 1)))
        self.log_interval = max(1, int(cfg_probe.get("log_interval", 500)))
        self.batch_size = int(cfg_probe.get("batch_size", 8))
        manifest = str(cfg_probe["manifest"])
        seg_sec = float(cfg_probe.get("segment_seconds", 15.0))

        ds = get_audio_wds(
            WebDatasetConfig(
                urls=manifest,
                sample_rate=int(sample_rate),
                segment_seconds=seg_sec,
                shuffle_size=int(cfg_probe.get("shuffle_size", 200)),
                resampled=True,
            )
        )
        self._dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=int(cfg_probe.get("num_workers", 1)),
            collate_fn=collate_fixed,
            drop_last=True,
        )
        self._it = iter(self._dl)

        charset_path = pathlib.Path(out_root) / "probe_charset.json"
        if charset_path.exists():
            self.charset = json.loads(charset_path.read_text(encoding="utf-8"))
        else:
            target = int(cfg_probe.get("vocab_bootstrap_samples", 1000))
            self.charset = self._bootstrap_charset(target)
            charset_path.write_text(json.dumps(self.charset, ensure_ascii=False), encoding="utf-8")
            self._it = iter(self._dl)

        self.vocab: Dict[str, int] = {c: i for i, c in enumerate(self.charset)}
        self.head = nn.Linear(int(encoder_dim), len(self.charset)).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=float(cfg_probe.get("lr", 1.0e-3)))
        self.ctc = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        self._buf: Dict[str, Any] = self._empty_buf()

    @staticmethod
    def _empty_buf() -> Dict[str, Any]:
        return {"loss_sum": 0.0, "n": 0, "last_logp": None, "last_texts": None}

    def _next_batch(self) -> Optional[Dict[str, Any]]:
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self._dl)
            try:
                return next(self._it)
            except StopIteration:
                return None

    def _bootstrap_charset(self, target: int) -> List[str]:
        texts: List[str] = []
        attempts = 0
        max_attempts = max(50, target // max(1, self.batch_size) * 3)
        while len(texts) < target and attempts < max_attempts:
            batch = self._next_batch()
            attempts += 1
            if batch is None:
                break
            for m in batch["meta"]:
                t = (m.get("text") if m else "") or ""
                t = t.strip()
                if t:
                    texts.append(t)
        return build_charset(texts)

    def step(self, model: nn.ModuleDict, step_idx: int, use_amp: bool) -> Optional[float]:
        if not self.enabled or step_idx % self.step_interval != 0:
            return None
        batch = self._next_batch()
        if batch is None:
            return None

        texts = [(m.get("text") if m else "") or "" for m in batch["meta"]]
        valid = [i for i, t in enumerate(texts) if t.strip()]
        if not valid:
            return None
        wav = batch["wav"][valid].to(self.device, non_blocking=True)
        texts = [texts[i].lower() for i in valid]

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                h0 = model["frontend"](wav)
                hE = model["encoder"](h0)  # (B, D, T')
        feats = hE.transpose(1, 2).float().contiguous()  # (B, T', D)

        target_ids: List[List[int]] = []
        for t in texts:
            ids = [self.vocab[c] for c in t if c in self.vocab]
            target_ids.append(ids)
        target_lens = torch.tensor([len(t) for t in target_ids], dtype=torch.long, device=self.device)
        if (target_lens == 0).any().item():
            return None
        if (target_lens > feats.size(1)).any().item():
            # CTC requires input_len >= target_len. Skip if any target longer than feats.
            return None

        logits = self.head(feats)  # (B, T', V)
        log_probs = logits.log_softmax(dim=-1)
        input_lens = torch.full((feats.size(0),), feats.size(1), dtype=torch.long, device=self.device)
        tcat = torch.tensor([i for ids in target_ids for i in ids], dtype=torch.long, device=self.device)
        loss = self.ctc(log_probs.transpose(0, 1), tcat, input_lens, target_lens)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self._buf["loss_sum"] += float(loss.item())
        self._buf["n"] += 1
        self._buf["last_logp"] = log_probs.detach()
        self._buf["last_texts"] = texts
        return float(loss.item())

    def maybe_emit(self, step_idx: int, wb: Any) -> None:
        if not self.enabled or step_idx == 0 or step_idx % self.log_interval != 0:
            return
        if self._buf["n"] == 0:
            return
        avg_loss = self._buf["loss_sum"] / self._buf["n"]
        wer_val = float("nan")
        if self._buf["last_logp"] is not None and self._buf["last_texts"] is not None:
            hyps = greedy_decode_ctc(self._buf["last_logp"], self.charset)
            try:
                from jiwer import wer as _wer
                wer_val = float(_wer(self._buf["last_texts"], hyps))
            except Exception:
                pass
        row = {
            "probe/ctc_loss": avg_loss,
            "probe/wer": wer_val,
            "probe/vocab_size": float(len(self.charset)),
        }
        if wb is not None:
            wb.log(row, step=step_idx)
        self._buf = self._empty_buf()

    def state_dict(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "head": self.head.state_dict(),
            "opt": self.opt.state_dict(),
            "charset": self.charset,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not self.enabled or not state:
            return
        if "charset" in state and state["charset"] != self.charset:
            return
        self.head.load_state_dict(state["head"])
        self.opt.load_state_dict(state["opt"])
