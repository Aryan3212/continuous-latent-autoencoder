# Static-analysis diagnosis

Run on commit `830b84d` via `uvx ruff`, `uvx vulture --min-confidence 60` (with
allowlist), `npx pyright`, and a hand-rolled intra-project import graph.

## TL;DR — what the tools found that the architectural review didn't

1. **Real bug — undefined names in `train.py`** (ruff F821, pyright
   `reportUndefinedVariable`):
   The legacy dense-JEPA `else` branch was partly removed. The current file
   references `use_global_local`, `num_views`, and `_dense_jepa_loss` at
   `train.py:736, 776, 784, 788, 821` — none of which are defined anymore.
   Reachable only if `num_globals < 1 or num_locals < 1`, which `train.py:533`
   raises on at startup — so this never fires at runtime but is dead, broken
   code that should be deleted.

2. **Real bug — missing `Optional` import in `losses/multires_stft.py`**
   (ruff F821, pyright `reportUndefinedVariable`):
   Lines 56 and 58 annotate `mask: Optional[torch.Tensor] = None` and
   `target_mags: Optional[List[torch.Tensor]] = None`, but `Optional` is
   never imported. Today this only errors if Python evaluates the annotation
   (would on `from __future__ import annotations` absent; the file has it,
   so it silently passes). Should be imported anyway.

## Confirmed dead symbols (delete candidates)

### Module-level

| Item | Source | Tool(s) |
| --- | --- | --- |
| `models/predictor.py` (whole file, ~45 lines) | not imported anywhere; no `__main__` | import-graph + vulture |
| `train.py:_primary_infonce` (~25 lines) | unused function | vulture, pyright |
| `utils/checkpoint.py:sha256_file` | unused function | vulture |
| `optim/lr_schedulers.py:get_last_lr` | unused method | vulture |
| `models/zipformer.py:ScalarMultiply` (class) | unused class | vulture |
| `models/zipformer_scaling.py:MaxEigLimiterFunction` | unused class | vulture |
| `models/zipformer_scaling.py:ScaleGrad` | unused class | vulture |
| `models/zipformer_scaling.py:SwooshLFunction` | unused class | vulture |
| `models/zipformer_scaling.py:SwooshLOnnx` | unused class | vulture |
| `models/zipformer_scaling.py:SwooshROnnx` | unused class | vulture |
| `models/zipformer_scaling.py:ScaledConv1d` / `ScaledConv2d` | unused functions | vulture |
| `models/zipformer_scaling.py:random_cast_to_half` | unused function | vulture |
| `models/zipformer_scaling.py:penalize_abs_values_gt` | unused function | vulture |

The vendored Zipformer tree dominates this list — consistent with Section C
of the review (encoder.py only uses 2 symbols from a ~3,800-line subtree).

### Local-scope dead vars

| Location | Item |
| --- | --- |
| `train.py:546` | `primary_temp` (retired InfoNCE knob) |
| `train.py:700, 708` | `has_mix` (retired mix path) |
| `train.py:671, 689` | `micro`, `i_mb` loop indices unused |
| `train.py:598` | `_encode` returns `hE, stats` that are never read |
| `train.py:92` | `_lejepa_invariance` unpacks `T` it never uses |
| `data/augment.py:202` | `B`, `C` unpacked from `wav.shape`, never used |
| `data/augment.py:222` | `device = wav.device` never read (the Python loop does `wav[i]` instead of building once) |
| `eval/eval_asr.py:171` | `targets` parameter shadowed/unused |
| `eval/run_probes.py:35` | `proc` capture unused |
| `models/decoder_generator.py:121` | `up_i` loop index unused |
| `models/zipformer.py:215, 749, 750` | `num_frames0`, `warmup_begin`, `warmup_end` unused |

### Unused imports (ruff F401 / pyright `reportUnusedImport`)

| File | Imports to drop |
| --- | --- |
| `data/augment.py:5-6` | `typing.List`, `typing.Optional`, `math` |
| `eval/eval_recon.py:6` | `typing.Any` |
| `eval/baselines.py:8,16` | `encodec`, `fairseq` (stubs never wired) |
| `models/encoder.py:4` | `typing.Tuple`, `typing.Dict` |
| `models/mhc.py:9` | `torch.nn.functional as F` |
| `utils/config.py:5` | `typing.Tuple` |
| `utils/logging.py:5` | `typing.Optional` |

## Dead config knobs (vulture confirmed)

- `data/augment.py:WaveAugConfig.reverb_prob` — declared, never read.
- `models/encoder.py:EncoderConfig.warmup_batches` — declared, never read.

## What static analysis can't see (still need the architectural pass)

These are *runtime-reachable but config-locked-off* — they look live to
ruff/vulture/pyright but the YAML never turns them on. From the earlier
review, in suggested deletion order:

1. **`optim/scaled_adam.py`** (223 lines) + `tests/test_scaled_adam_parity.py`
   (147 lines) + the `if ocfg["kind"] == "scaled_adam":` branch in `train.py`.
2. **`optim/lr_schedulers.py`** (Eden / Eden2, 116 lines) + the two branches in
   `train.py` that select them.
3. **`latent_noise` plumbing** (`_latent_noise_sigma`, σ parameter on
   `_decode`, the `latent_noise` config block).
4. **Mix / `mix_recon` / `primary` paths** (`MixConfig`, `maybe_mix_pair`,
   `_primary_infonce` already flagged above, `batch_b` double-iteration,
   `wav_mix`, `wav_tgt`, `mixed_mask`, `primary_idx`, `snr_db_vals`,
   `l_jepa_mix`, `l_stft_mix`).
5. **Legacy dense JEPA recipe** (`_dense_jepa_loss`, the broken `else` branch
   above, `feature_mask` config + `apply_feature_mask`).
6. **Decoder latent-stats normalisation** (`set_latent_stats`, `latent_mean` /
   `latent_var` buffers, `latent_norm` branch, `latent_stats_path` reader in
   `train.py`).

After deleting (1)–(6), re-run all three tools — the orphans they cascade
into are the second-pass deletion list. The vendored Zipformer functions
above will likely show up much more aggressively because their last live
import goes away too.

## Files generated by this audit

- `.static-analysis/ruff.json` — full ruff JSON (4239 lines)
- `.static-analysis/ruff-key.txt` — F401/F811/F841/F821 only (18 findings)
- `.static-analysis/vulture.txt` — 70%-confidence pass (7 findings)
- `.static-analysis/vulture-filtered.txt` — 60% confidence + allowlist
- `.static-analysis/vulture-clean.txt` — above, framework methods stripped
- `.static-analysis/pyright.json` — full pyright output
- `.static-analysis/pyright-clean.txt` — Unused/Undefined/Unreachable only
- `.static-analysis/import-graph.txt` — intra-project import map
- `.static-analysis/vulture-allowlist.py` — framework-magic suppressions
- `.static-analysis/pyrightconfig.json` — pyright project config
