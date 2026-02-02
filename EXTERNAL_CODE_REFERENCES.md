# External code references (cloned repos)

This file records where the reference implementations live inside the cloned repos that are vendored into this project directory, along with the repo commit hash currently checked out.

## `icefall/` (Zipformer + ScaledAdam)

- Repo commit: `0904e490c5fb424dc5cb4d14ae468e4d32a07dc4`
- Zipformer encoder:
  - `icefall/egs/librispeech/ASR/zipformer/zipformer.py:37` (imports `BiasNorm`, `Swoosh*`, etc.)
  - `icefall/egs/librispeech/ASR/zipformer/zipformer.py:53` (`class Zipformer2`)
  - `icefall/egs/librispeech/ASR/zipformer/scaling.py:425` (`class BiasNorm`)
  - `icefall/egs/librispeech/ASR/zipformer/scaling.py:1401` (`class SwooshL`)
  - `icefall/egs/librispeech/ASR/zipformer/scaling.py:1475` (`class SwooshR`)
- ScaledAdam optimizer:
  - `icefall/egs/librispeech/ASR/zipformer/optim.py:153` (`def scaling_step`)
  - `icefall/egs/librispeech/ASR/zipformer/optim.py:257` (`class ScaledAdam`)
  - `icefall/egs/librispeech/ASR/zipformer/optim.py:841` (`class Eden` LR scheduler; suggests `base_lr = 0.04` for ScaledAdam)
  - `icefall/egs/librispeech/ASR/zipformer/train.py:1343` (ScaledAdam + Eden wiring)

## `lejepa/` (LeJEPA + SIGReg)

- Repo commit: `c293d291ca87cd4fddee9d3fffe4e914c7272052`
- Minimal SIGReg (paper snippet-style) implementation:
  - `lejepa/MINIMAL.md:57` (`class SIGReg`)
- LeJEPA objective pseudocode:
  - `papers/lejepa.pdf` Algorithm 1 (SIGReg) and Algorithm 2 (LeJEPA) (used as the primary spec)
- Library components used to build SIGReg-style objectives (distribution tests + slicing):
  - `lejepa/lejepa/univariate/epps_pulley.py:1` (Epps–Pulley test)
  - `lejepa/lejepa/multivariate/slicing.py:1` (`SlicingUnivariateTest`)
  - `lejepa/README.md:101` (example composing univariate + multivariate slicing)

## `RAE/` (Representation Autoencoder)

- Repo commit: `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`
- Core model:
  - `RAE/src/stage1/rae.py:12` (protocol + imports)
  - `RAE/src/stage1/rae.py:17` (`class RAE`)

## `mHC-manifold-constrained-hyper-connections/` (mHC)

- Repo commit: `088310e54680375e7647f02c811874f7c11a690d`
- Sinkhorn projection and mHC update logic:
  - `mHC-manifold-constrained-hyper-connections/hyper_connections/hyper_connections_mhc.py:45` (`def sinkhorn_log`)
  - `mHC-manifold-constrained-hyper-connections/hyper_connections/hyper_connections_mhc.py:96` (`class HyperConnections`)
  - `mHC-manifold-constrained-hyper-connections/hyper_connections/hyper_connections.py:50` (`def sinkhorn_log` baseline/HC impl)

## Paper hyperparameter tables (CALM)

- `papers/continuaudiollm.pdf`:
  - Table 13: VAE hyperparameters (frame rate 12.5Hz for speech; latent dim 32; KL weight 0.01; cosine LR; LR 8e-4).
  - Table 14: training hyperparameters (AdamW β1=0.9 β2=0.95; cosine LR; learning rate 5e-5–2e-4 range; long audio sample lengths).
