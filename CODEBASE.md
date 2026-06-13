# Codebase map (quick reference)

A deterministic continuous-latent speech autoencoder. Waveform in (16 kHz) →
strided conv frontend → Conformer encoder → continuous latents `z` → conv/FiLM
decoder back to waveform, trained jointly with reconstruction + JEPA + SIGReg.

## Core pipeline

All modules take their pydantic `*Cfg` from `utils/schema.py` directly (no
mirror dataclasses). Internal tensor layout is `(B, T, D)`; external module
interfaces use channels-first `(B, D, T)` as noted.

- `models/frontend_conv.py` — `ConvFrontend`: strided Conv1D stack with
  GroupNorm+GELU. `(B, 1, T)` → `(B, C, T')`. Stride product 1280 → ~12.5 Hz
  tokens at 16 kHz.
- `models/encoder.py` + `models/conformer.py` — `Encoder`: 1×1 in-proj then a
  stack of `ConformerLayer`s (macaron FFN → MHSA+RoPE → conv → FFN → LN,
  `F.scaled_dot_product_attention`). `(B, C, T')` → `(B, D, T')`.
- `models/mhc.py` — `MHCWrapper` + `sinkhorn_log`: Manifold Hyper-Connections,
  applied on selected encoder layers (residual streams `(S, B, T, D)`). Kept
  for an upcoming on/off ablation (see `LAB_NOTEBOOK.md`).
- `models/projector.py` — `Projector`: per-frame MLP + BatchNorm1d head.
  `(B, D, T')` → `(B, P, T')`. Decouples loss-space (JEPA + SIGReg act here)
  from representation-space (`z`, which the decoder and probes consume).
- `models/decoder_generator.py` — `WaveformDecoder`: ConvTranspose upsample
  stack with FiLM ResBlocks conditioned on the latent. `(B, D, T')` → `(B, 1, T)`.
- `models/sigreg.py` — `SIGReg` (Epps–Pulley sliced univariate Gaussianity
  test). `forward` returns a bare scalar. Single-GPU only.
- `losses/multires_stft.py` — `MultiResSTFTLoss`: multi-resolution STFT
  reconstruction (spectral convergence + log-magnitude).

## Training

- `train.py` — single entrypoint and training loop. Objective:
  `stft_weight·L_stft + jepa_weight·L_jepa + sigreg_weight·L_sig` (+ optional
  L1 wav term). `L_jepa` is the V-JEPA global/local dense loss
  (`l_global` + `l_predict` + `context_weight·l_context`); `l_global ≡ 0` when
  `num_globals == 1`. `L_sig` is frame-level SIGReg on the projector output.
  Validation computes `val_stft` only; the val dataloader is built once at
  startup.
- Configs: `configs/exp0.yaml` (cloud, ~6M params, d_model 192) and
  `configs/local_6gb.yaml` (local 6 GB-VRAM PC, smaller). Both are valid
  full configs, not overrides.

## Config system

`utils/schema.py` (pydantic, `extra="forbid"`) is the single source of truth.
`utils/config.py`: `load_config(path)` parses YAML → `Config`;
`apply_overrides(cfg, ["a.b=c", ...])` applies dotted CLI overrides. Both
return the pydantic model; call sites use attribute access. `DatasetConfig`
in `data/dataset.py` is a runtime object, intentionally not YAML-mirrored.

## Evaluation

Frozen-encoder probes + reconstruction metrics.

- `eval/eval_asr.py` — CTC ASR probe (small head over frame features), WER.
- `eval/eval_cls_probe.py` — pooled-embedding MLP probe (`--label_key` selects emotion/gender).
- `eval/eval_recon.py` — waveform reconstruction metrics (STFT etc.).
- `eval/common.py` — shared frame-feature extraction.
- `eval/run_probes.py` — orchestrates the three probes (direct schema access).
- `eval/run_all.py` — CLI: recon + all enabled probes in one call.

## Data prep (`clae_data/`)

One unified package replacing the old scattered prep scripts. Per-source
adapters (`adapters/`: openslr53, bengaliai_speech, regspeech12, indicvoices,
subak_ko, shrutilipi, kathbath — each a `DatasetAdapter` with
`download()` + `iter_records()`) → `pack.py` writes 16 kHz mono FLAC under
`audio/<dataset>/` plus four JSONL manifests (train/val/asr_probe_{train,val})
→ `push.py`/`fetch.py` move the packed layout to/from HF Hub as raw-file blob
storage. Driven by `python -m clae_data` and the `Makefile`. Credentials live
in the gitignored `clae_data/_creds.py` (`_creds.example.py` is the template);
on a fresh GPU VM, `./setup.sh` generates it from `.env` (template:
`.env.example`) and runs deps → fetch → train in one shot (`--no-train` to
stop after the data fetch). Only `HF_TOKEN` is needed for training; Kaggle
keys only for `make pack-and-push` on a prep instance.

## Data loading + augmentation

- `data/dataset.py` — `AudioDataset` (JSONL manifests; relative
  `audio_filepath` resolved via `resolve_manifest_root`: manifest dir or one
  level up for the packed `<root>/manifests/` layout), `collate_fixed`.
- `data/augment.py` — waveform augment (noise/lowpass/gain/clip) and
  frame/waveform chunk masking for JEPA local views.

## Scripts (`scripts/`)

- `get_param_count.py` — four-block parameter breakdown for a config.
- `check_rank.py` — latent participation-ratio / effective-rank probe.
- `verify_experiment.py` — end-to-end forward/decode sanity check + plots.
- `reconstruct_audio.py` — encode/decode audio files through a checkpoint;
  writes `_orig`/`_recon` WAV pairs + per-file STFT/L1 numbers.
- `visualize_latents.py` — PCA/UMAP latent-space plot.
- `smoke_encoder_mhc.py` — encoder+mHC forward smoke test.
- `test_mhc.py` — standalone mHC wrapper check.

## Tests (`tests/`)

`test_sigreg.py`, `test_multires_stft.py` — run with `uv run pytest tests/`.

## Reference implementations (`reference-implementations/`)

Slim, in-tree: single-file refs (`lejepa_*`, `mhc_*`, `zipformer2_*`) +
`*-REIMPLEMENTATION_NOTES.md` + a README. Full vendored upstream repos
(`vjepa2`, `RAE-main`, `le-wm`) were moved out of the repo to
`../reference-implementations-archive`.

## Other docs

- `README.md` — setup, running things, cloud one-command flow, credentials.
- `CHANGELOG.md` — dated log of major changes.
- `LAB_NOTEBOOK.md` — experiment log + open research decisions.
- `docs/` — live research notes (log interpretation, experiment plan, SIGReg
  notes).
