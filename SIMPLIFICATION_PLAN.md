# Simplification Plan — Remaining Work

Most of the original plan was executed in a single subagent-driven pass on
branch `simplification` (one commit, off `main` so the Zipformer-era main is
preserved). C3–C5, C7 were completed in a follow-up pass (2026-05-22).

For context on what was already done, see the git history on the
`simplification` branch and `.static-analysis/DIAGNOSIS.md`.

## Conformer implementation check (2026-05-22)
Verified against official torchaudio source. Structure is correct:
macaron-Conformer order (FFN₁ → MHSA → Conv → FFN₂ → FinalLN), pre-norm
throughout, residual connections identical to official. Two intentional
divergences from the reference (not bugs): rotary positional embeddings in
place of no-PE, and `F.scaled_dot_product_attention` in place of
`nn.MultiheadAttention`. Key-padding-mask convention (True = ignore) is
consistent between the mask builder and SDPA additive-mask path. No changes
needed.

---

## Caveats from the simplification pass (act on these first)

### C1 — Old checkpoints will not load - IGNORE already have separate branch
The Zipformer → Conformer rewrite (`models/conformer.py` + rewritten
`models/encoder.py`) renames every encoder parameter. Any `.pt` saved against
`Zipformer2EncoderLayer` is incompatible. **Restart training from scratch on
this branch.** No state_dict shim is provided.

### C2 — ~~Manifest paths are placeholders~~ PARTIALLY DONE
`configs/exp0_20pct.yaml` deleted; `configs/exp0.yaml` is the only config remaining.
Manifest paths in `exp0.yaml` are still placeholder paths (`data/manifests/train.jsonl` etc.)
— correct them to real paths on the training box before the first run.

### C3 — ~~Two configs will fail the new pydantic schema~~ DONE
`configs/exp0_20pct_merged.yaml` and `configs/exp0_test.yaml` deleted.
`configs/exp0.yaml` (canonical) and `configs/exp0_20pct.yaml` (valid override) remain.

### C4 — ~~`scripts/pack_webdataset.py` is now orphan~~ DONE
Deleted.

### C5 — ~~Style nit in `train.py:455` (semicolon)~~ DONE
Split into two lines.

### C6 — `eval/run_probes.py:35` `proc` left unused 
`subprocess.run(...)` assigned to `proc` and never read. The §2.3 subagent
left it deliberately so a future error handler can reach the process result;
revisit if you decide you want the cleaner form.

### C7 — ~~`CHANGELOG.md` was not updated~~ DONE
Entry added for the simplification branch (2026-05-22).

---

## §3.8 — ~~call-site migration of config access~~ DONE (2026-05-22)

`load_config` and `apply_overrides` now return `Config` (pydantic model).
All `cfg["..."]` / `cfg.get("...")` sites in `train.py`, `eval/`, and
`scripts/` converted to `cfg.attr` attribute access. Type casts (`int()`,
`float()`, `bool()`) and the dict-key filters for projector/sigreg dropped.
`save_checkpoint` / `save_run_metadata` serialize via `cfg.model_dump()` so
checkpoint dicts remain loadable without pydantic.

`eval/run_probes.py`: emotion/gender probe code updated with `exp_cfg.eval.*`
attribute access; those probes print a skip notice when enabled until their
manifests are added to the schema.

`scripts/reconstruct_sample.py` and `scripts/count_params.py` are exempt —
they load cfg from a checkpoint dict or raw YAML respectively, not via
`load_config`.

Verify on training box: run `python train.py --config configs/exp0.yaml
data.train_manifest=... data.val_manifest=... --max_steps 1` to catch any
AttributeError mismatches before a full run.

---

## §3.3 — MHC ablation (decision-deferred experiment) - keep i'll ablate using this

MHC machinery (`models/mhc.py`, the wrapper plumbing in `models/encoder.py`,
the `model.encoder.mhc` config block) is **kept on this branch** but the
ablation decision is still open.

**Plan:**
- Run two training jobs side by side (same seed, same data):
  - MHC on:  `model.encoder.mhc.enabled=true`  (current config).
  - MHC off: `model.encoder.mhc.enabled=false` (or `num_streams=1`).
- Compare ASR-probe WER, JEPA loss curves, SIGReg variance.
- Decide:
  - **No clear win for MHC** → delete `models/mhc.py`, strip the
    `mhc_wrappers` / `_mhc_layers` / `_apply_per_stream` machinery from
    `models/encoder.py`, drop the `MHCCfg` from `utils/schema.py`, remove the
    `mhc:` block from every config.
  - **MHC wins** → document what it's winning on so the keeper rationale is
    in the repo.

No code changes until the experiment runs.

---

## §3.6 — Eval surface (decision-deferred) - keep currently

`eval/eval_emotion.py`, `eval/eval_gender.py`, `eval/extract_embeddings.py`
are config-disabled but still on disk. Left in place during the pass — they
work as written, just unused.

`iter_embeddings` and `iter_frame_features` in `eval/common.py` are ~90%
duplicate. Could be unified with a `pool: bool` flag and one helper. Left
for later.

**Decision needed:** keep these eval entrypoints (re-enable later) or delete
once the project commits to the inline-probe-only path?

---

## §3.7 — ~~GAN code~~ DELETED (2026-05-22, later pass)

GAN training path removed entirely (discriminators, adaptive weight, two
`torch.autograd.grad` calls, hinge loss, schema block, configs). Recoverable
from git history at commit before the `train.py` audit pass if reintroduced.
Reason: not on the near-term roadmap, and the two `torch.autograd.grad` calls
are DDP/`torch.compile`-hostile.

## Bring-back-later (2026-05-22, train.py audit pass)

### Online/offline probe

The inline CTC probe (`eval/inline_probe.py`) was deleted along with its
config block and call sites in `train.py`. The eval-on-save subprocess
orchestrator block in `train.py` (which shuffled the model to CPU, shelled
out to `eval/run_probes.py`, and saved `best_asr.pt` / `best_composite.pt`)
was also removed.

**Why it's gone:** sanity-check probing should be either
(a) cleanly inline and torch.compile/DDP-safe — the previous detached-head
implementation wasn't, or
(b) a separately-launched offline process — which `eval/eval_asr.py` already
supports (call with `--config`, `--ckpt`, `--train_manifest`, `--dev_manifest`).

**To bring back:** decide which shape you want. If inline: a clean
nn.Module-only head that participates in the main forward (don't toggle
requires_grad inside the step). If offline: a launcher script that finds the
latest `last.pt` and runs `eval/eval_asr.py` against it on a cadence,
out-of-process.

### Best-checkpoint tracking

Only `last.pt` is saved. `best_asr.pt` / `best_composite.pt` are gone because
the metric used to gate them came from the deleted eval-on-save block. When
the probe comes back, re-add best-tracking *only* if you'll actually use those
checkpoints in downstream work; otherwise rank from logs and copy by hand.

### CodeCarbon emissions tracking

Removed. Re-add if you ever want it back — it's ~10 lines.

---

## §5 — ~~Static-analysis cadence~~ DONE (2026-05-22)

Reports regenerated at `.static-analysis/ruff.txt` and `.static-analysis/vulture.txt`.
Pyright not re-run (npx/Node required separately).

**Ruff summary (14 findings, all pre-existing):**
- `eval/run_probes.py:38 F841` — `proc` unused (C6, deliberate)
- `eval/common.py:123 ARG001`, `eval/eval_asr.py:171 ARG001` — unused args (legacy)
- `models/mhc.py:43 ARG002`, `models/sigreg.py:103 ARG002` — unused args (legacy)
- `ERA001` scattered in losses/, models/ — commented-out code, pre-existing
- `RUF046` in `data/` — redundant int casts, pre-existing

**Vulture summary (real signals, after filtering nn.Module.forward false positives):**
- `train.py:57 _pool_utt`, `train.py:141 _pool` — unused helper functions
- `eval/eval_asr.py:171 targets` — unused variable (100% confidence)
- `eval/run_probes.py:38 proc` — C6 deliberate survivor
- `utils/schema.py model_config` — pydantic metaclass field, false positive

Re-run commands (from README.md Static analysis section) when ready for the
next cleanup pass.
