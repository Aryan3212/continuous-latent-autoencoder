# RESEARCH SUMMARY: CONTINUOUS LATENT AUDIO ENCODER (CLAE)
# DATE: 2026-04-21
# RE: ARCHITECTURAL EVOLUTION AND EXPERIMENTAL FINDINGS

## 1. OVERVIEW
This report summarizes the iterative development of a Continuous Latent Autoencoder (CLAE) designed to bridge the gap between high-fidelity neural audio codecs and robust self-supervised representation learners.

## 2. ARCHITECTURAL FOUNDATIONS
*   **ENCODER:** We moved away from standard Conformer stacks to a Zipformer2-based architecture. To improve gradient flow in deep layers, we integrated Manifold Hyper-Connections (mHC), allowing for more expressive feature extraction by constraining the residual streams to the manifold.
*   **DECODER:** We transitioned from a standard bottleneck VAE to a Representation Autoencoder (RAE) framework. This involved increasing the latent dimensionality (256-dim) and injecting Gaussian noise during training.
*   **HYPOTHESIS:** We found that tight bottlenecks "strangled" semantic feature learning. High-dimensional latents combined with "RAE noise" created a smoother data manifold, serving both reconstruction and downstream utility.

## 3. SELF-SUPERVISED OBJECTIVES
We iterated through several pretext tasks to induce semantic structure in the latent space:
*   **LeJEPA (Joint-Embedding Predictive Architecture):** Initially, we used cosine similarity to align clean and masked views.
*   **INFONCE (CONTRASTIVE LEARNING):** Early experiments showed simple similarity tasks plateaued at random chance. We pivoted to InfoNCE with in-batch negative sampling, which successfully drove discriminative feature learning.
*   **MASKED AUDIO RESTORATION (MAR):** We evolved the decoder task from simple identity mapping to "inpainting." By penalizing reconstruction only on masked regions, we forced the encoder to capture global context to recover the missing signal.

## 4. LATENT SPACE REGULARIZATION (SIGREG)
A primary challenge was preventing "dimensional collapse." We implemented Statistical Isotropic Gaussian Regularization (SIGReg):
*   **STOCHASTIC WHITENING:** We applied the Epps-Pulley test for Gaussianity using trapezoidal quadrature.
*   **HIERARCHICAL DECOMPOSITION:** We refined the regularization to test utterance-level means and per-frame residuals separately. This prevented the global mean from drifting while ensuring local frames remained isotropic.
*   **HYPOTHESIS:** This dual-level regularization is critical for speech, as it preserves speaker-identity (global) while regularizing phonemic variations (local).

## 5. AUDIO FIDELITY & GAN DYNAMICS
To achieve high-perceptual detail without "phasiness":
*   **ADAPTIVE GAN WEIGHTING:** We implemented VQGAN-style adaptive balancing. The GAN loss weight is dynamically scaled based on the ratio of gradients from the STFT reconstruction loss at the decoder's last layer.
*   **STABILIZATION:** We identified that adversarial training in this architecture is numerically sensitive. We forced GAN operations to float32 and implemented separate GradScalers for the Generator and Discriminator.
*   **SPECTRAL MASKING:** We moved masking from the time-domain to the frequency-domain (STFT magnitudes), eliminating high-frequency artifacts at mask boundaries.

## 6. ENGINEERING & OPTIMIZATION
*   **TORCHDYNAMO:** To enable `torch.compile`, we moved stochastic whitening logic to the custom autograd backward pass, preventing graph divergence.
*   **MEMORY MANAGEMENT:** We utilized `expandable_segments` and selective mixed-precision (AMP) to fit heavy evaluation probes (ASR/Emotion/Gender) into VRAM during training checkpoints.

## 7. CURRENT STATUS
The model now demonstrates a stable training curve with decreasing InfoNCE loss and reconstruction error. The latent space maintains high "Participation Ratio" (effective rank), and the decoder is robust to significant acoustic distortions (1.0 probability of noise/filtering during training).
