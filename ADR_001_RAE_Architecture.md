# Architectural Decision Record (ADR)

## Decision 001: Adoption of RAE-Style Architecture

### Context
We initially designed a standard Autoencoder with a tight, deterministic bottleneck (32-dim) regularized by LeJEPA and SIGReg. However, review of the LeJEPA paper and the "Representation Autoencoder" (RAE) paper revealed a conflict:
1.  **LeJEPA Constraint**: Requires high-dimensional embeddings (e.g., 256+) to effectively learn robust features and avoid collapse.
2.  **Bottleneck Constraint**: Our bottleneck was 16-32 dims, strangling the LeJEPA objective.
3.  **RAE Insight**: High-fidelity reconstruction and generation are *better* served by high-dimensional, semantic latents (no bottleneck) than compressed VAE latents.

### Decision
We are pivoting to a **Continuous RAE (Representation Autoencoder)** architecture.

1.  **Removal of Low-Dim Bottleneck**:
    *   The `Bottleneck` module (projection to 32-dim) is removed.
    *   The latent space $z$ is now the **Encoder Output `hE`** (256-dim).
    *   This preserves the rich semantic information LeJEPA needs.

2.  **Loss Application**:
    *   **LeJEPA + SIGReg**: Applied directly to the 256-dim `hE`.
    *   **Reconstruction**: Decoder takes `hE` and reconstructs waveform.

3.  **Noise-Augmented Decoding**:
    *   Following the RAE paper's "Decoder Trick", we inject Gaussian noise into `hE` before decoding during training.
    *   This ensures the decoder is robust to the slight imperfections of future diffusion models (or just general robustness).

### Implications
*   **Compression Rate**: This model is no longer a high-compression codec (256 dims @ 50Hz is high bitrate). It is a **Foundation Model Representation Learner**. Compression can be added later as a separate stage (e.g., separate VQ layer or Diffusion model).
*   **Decoder**: We stick with the Convolutional/FiLM decoder for now, as it is efficient for audio.

### Status
*   [x] Config updated to enable `latent_noise`.
*   [x] Config updated to set `latent_dim` = `d_model` (256).
*   [ ] `train.py` modified to bypass bottleneck projection.
