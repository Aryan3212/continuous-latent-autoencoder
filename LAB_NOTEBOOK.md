# Lab Notebook

## Experiment Log

| Date | Run ID | Hypothesis | Outcome | Notes |
|------|--------|------------|---------|-------|
| 2026-02-10 | exp0_sweep | Hyperparameter sweep for baseline | Pending | Sweeping lr, d_model, latent_dim, loss weights |
| 2026-02-11 | n/a | Numerical instability in STFT loss during silence | Resolved | Increased `logmag_eps` to 1e-3 in `losses/multires_stft.py` and `configs/exp0.yaml` to prevent Spectral Convergence explosion on near-zero targets. |


## Human log
1. We are not doing latent normalization that was introduced in the RAE paper, we are also not using the Vision Transformer(ViT)-like decoder. Or audio transformer like setup, but maybe we can explore that later. The paper added noise to the latents from the pretrained encoder, which we are adding in hopes that the noise helps the decoder be invariant to slight perturbations to the features and can create smooth data manifold that has a semantic understanding of the data.
2. RAE requires high dimensionality for reconstruction we are preserving that here.
3. We stick with the Convolutional/FiLM decoder for now, as it is efficient for audio. Not ViT as previously mentioned.
