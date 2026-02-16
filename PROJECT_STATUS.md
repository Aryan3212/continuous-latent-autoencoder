# Project Status & Documentation (Consolidated)

*Last Updated: 2026-02-15*

This document consolidates the project's architecture, codebase structure, dataset workflows, and current implementation status. It supersedes previous individual documentation files.

---

## 1. Project Overview

**Continuous Latent Audio Autoencoder**

A foundation model for audio representation learning, moving away from discrete codes (VQ-VAE) or tight bottlenecks (VAE) towards a **Continuous Representation Autoencoder (RAE)**.

-   **Goal**: Learn high-dimensional, semantic, and robust audio representations suitable for downstream generation (Stage 2 Diffusion).
-   **Key Mechanism**: Instead of compressing dimensions (e.g., 32-dim), we keep dimensions high (e.g., 256-dim) but regularize the space using **LeJEPA** (predictive learning), **SIGReg** (isotropy), and **Noise Injection** (smoothness).
-   **Input/Output**: 16kHz Waveform.

---

## 2. Architecture: Continuous RAE

### Core Philosophy (Pivot from VAE)
We initially considered a "Compressed Autoencoder" (VAE-like). However, to support **LeJEPA** (which needs high dimensionality to avoid collapse) and **RAE** findings (which show high-fidelity generation requires semantic latents), we pivoted:
-   **Latent Space**: The latent $z$ is the direct output of the Encoder $h_E$ (e.g., 256 dimensions). No projection down to 16/32 dimensions.
-   **Regularization**:
    1.  **LeJEPA**: Forces the encoder to map augmented views (noise, low-pass) to the same latent as the clean view.
    2.  **SIGReg**: explicit regularization to keep the latent covariance isotropic (preventing dimensional collapse).
    3.  **Latent Noise**: Gaussian noise is injected into $z$ before decoding (during training) to force the decoder to treat points as regions.

### Module Breakdown

#### 1. Frontend (`models/frontend_conv.py`)
-   **Input**: Raw waveform `(B, 1, T)`.
-   **Stack**: Strided Conv1D $\to$ GroupNorm $\to$ GELU.
-   **Output**: Low-level feature tokens (~12.5 Hz).

#### 2. Encoder (`models/encoder.py`)
-   **Backbone**: **Zipformer2** (from Icefall), an efficient transformer variant.
-   **MHC**: **Manifold-Constrained Hyper-Connections** (optional) to improve gradient flow and representation mix.
-   **Status**: Fully trainable (unlike the frozen Image RAE).

#### 3. Bottleneck (`models/encoder.py`)
-   **Type**: `identity` (RAE mode) or `deterministic` with Norm.
-   **Normalization**: LayerNorm or RMSNorm to keep scales reasonable.

#### 4. Decoder (`models/decoder_generator.py`)
-   **Type**: **WaveformDecoder** (Conv1D / HiFi-GAN style).
-   **Conditioning**: FiLM (Feature-wise Linear Modulation) layers modulate the residual blocks with the latent $z$.
-   **Latent Norm**: Optional statistics-based normalization (RAE style) if the encoder was frozen (currently disabled as we train end-to-end).

#### 5. Losses
-   **Reconstruction**: Multi-Resolution STFT Loss + L1 Waveform Loss.
-   **LeJEPA**: MSE between Clean Latent $z_{clean}$ and Augmented Latent $z_{aug}$.
-   **SIGReg**: Regularization on $z$ covariance.
-   **GAN (Optional)**: Multi-Period (MPD) and Multi-Scale (MSD) discriminators.

---

## 3. Implementation Status

### Completed Features
-   [x] **Zipformer2 Encoder**: Ported from Icefall.
-   [x] **MHC**: Integrated into Encoder.
-   [x] **ScaledAdam**: Ported optimizer from Icefall.
-   [x] **SIGReg**: Ported LeJEPA Algorithm 1 (Epps-Pulley test).
-   [x] **RAE Decoder**: FiLM conditioning and Latent Noise injection.
-   [x] **GAN**: Discriminators and losses wired in `train.py`.
-   [x] **Probes**: ASR, Emotion, and Gender evaluations integrated.

### Current Known Issues
-   **OOM**: The default batch size (8) in `exp0.yaml` causes CUDA OOM on the current 5.6GB VRAM GPU. **Action**: Reduce batch size to 2 or 4.
-   **Reference Code**: Vendored code in `icefall/`, `lejepa/`, `RAE/` is for reference only; core modules are self-contained in `models/`.

---

## 4. Dataset Workflow

### Source
-   **BengaliAI Speech Dataset** (Kaggle).
-   Location: `data/bengaliai_speech/train_mp3s`.

### Workflow
1.  **Split**: `scripts/create_dataset_splits.py` creates `train`/`val`/`test` JSONL manifests.
2.  **Audit**: `scripts/audit_datasets.py` scans for silence, clipping, and duplicates.
3.  **Clean**: `scripts/create_clean_manifest.py` filters the manifests based on the audit report.

### Expansion
To add new data (e.g., LibriSpeech), process it in isolation (Split $\to$ Audit $\to$ Clean) before merging JSONL files. Just merging the train files is enough.

---

## 5. Experimentation Guide

### Running Experiments
1.  **Baseline (Exp0)**:
    ```bash
    uv run python train.py --config configs/exp0.yaml data.train_manifest=...
    ```
2.  **Evaluation**:
    ```bash
    uv run python eval/run_all.py --config configs/exp0.yaml --ckpt ...
    ```

### Diagnostics
-   **Latent Collapse**: Monitor `z_std` and `z_var_min` in WandB. If `z_std` $\approx$ 0, the model has collapsed.
-   **Silence/Instability**: If loss explodes, check for silence in the dataset (mostly fixed by `audit_datasets.py` and `logmag_eps` fix).

---

## 6. Codebase Map

| Directory | Content |
| :--- | :--- |
| `configs/` | YAML configs (`exp0.yaml`, `calm_like_exp0.yaml`). |
| `data/` | Dataset loading (`dataset.py`), augmentation (`augment.py`), and mining. |
| `eval/` | Probes (`eval_asr.py`) and reconstruction eval (`eval_recon.py`). |
| `losses/` | STFT, LeJEPA, SIGReg losses. |
| `models/` | Core architecture (`encoder.py`, `decoder_generator.py`, `zipformer.py`). |
| `optim/` | `ScaledAdam` and `Eden` schedulers. |
| `scripts/` | Utilities (`audit_datasets.py`, `smoke_*.py`). |
| `train.py` | Main training loop. |

---

## 7. External References

We rely on several key papers and repositories:
-   **Icefall (Zipformer)**: Source of our Encoder and Optimizer.
-   **LeJEPA**: Source of our regularization and predictive objective.
-   **RAE (Representation Autoencoder)**: Source of our "No Bottleneck" + "Latent Noise" philosophy.
-   **mHC**: Source of the hyper-connection mechanism in the encoder.

See `EXTERNAL_CODE_REFERENCES.md` (now outdated/archived) for specific line numbers in vendored repos.
