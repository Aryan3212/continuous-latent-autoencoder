# Plan: Decoder design (RAE-inspired)

Target: align the decoder design with the Representation Autoencoder (RAE) paper/code ideas, rather than an ad-hoc vocoder-ish ConvTranspose stack.

## References to pin

- RAE paper summary: `paper-summaries/rae.md`
- RAE reference code:
  - `RAE/src/stage1/rae.py:17` (`class RAE`)
  - Key mechanisms to mirror:
    - latent normalization stats (`latent_mean`, `latent_var`),
    - latent noising during training (`noise_tau`),
    - robust decoding from noisy latents.

## What “correct” means (local spec to write into `UNCERTAINTIES.md`)

1. Decoder is trained to reconstruct from *representation latents* and is robust to latent noise.
2. Latents are centered/normalized consistently (optional but recommended).
3. Noise injection to latents is explicit, controlled, and documented (and can be scheduled).

## Step-by-step implementation plan

1. Decide the exact “RAE mapping” to 1D audio
   - RAE is image-token based; we need to decide whether we:
     - keep a 1D waveform decoder but adopt RAE’s *training mechanics* (normalization + noise), or
     - implement a token-decoder that predicts a higher-rate acoustic representation (e.g., STFT bins) and then invert to waveform.
2. Implement RAE-style latent normalization hooks
   - Add utilities to compute and save latent stats over a manifest:
     - `mean` and `var` for `z` (and possibly for `hE`) over the training set.
   - Add a config toggle to enable normalization in the decoder path.
3. Implement RAE-style latent noise (decoder-side)
   - Replace the current `sigma ~ Uniform(0, sigma_max)` mechanism with an RAE-like `noise_tau` control:
     - per-sample random noise sigma derived from `noise_tau`,
     - optionally keep the current uniform scheme as an alternative.
   - Ensure noise is applied to decoder input only (consistent with your existing Exp2 split).
4. Revisit decoder architecture
   - If we keep the current generator:
     - replace linear interpolation conditioning with either a learned upsampler or nearest/linear with a clear alignment policy.
   - If we change topology:
     - document a direct link to RAE’s `GeneralDecoder` idea (transformer-style decoder) and implement an audio-appropriate analogue.
5. Validation
   - Unit tests:
     - length handling (`target_len`) across multiple upsample settings,
     - gradients flow through decoder to `z`.
   - Smoke script:
     - run a forward/backward step with latent noise enabled.
6. Update `UNCERTAINTIES.md`
   - Replace “vocoder-ish + FiLM uncertain points” with the chosen RAE-inspired decoder spec and references.

