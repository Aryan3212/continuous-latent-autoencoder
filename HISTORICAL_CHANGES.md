# Historical Changes of Continuous Latent Autoencoder

This document provides a detailed log of the evolution of the Continuous Latent Autoencoder (CLAE) project, including changes made in each commit, their expected outcomes, and the underlying research hypotheses.

---

## [8d7ea73] init
**Original Message:** init

### **Changes Made**
*   **Encoder:** Ported a **Zipformer2** stack and integrated **Manifold Hyper-Connections (mHC)** to improve gradient flow and feature expression.
*   **Decoder:** Adopted a **Representation Autoencoder (RAE)** design, adding latent normalization and noise injection for robustness.
*   **Objectives:** Implemented a multi-stage curriculum including **Multi-Resolution STFT reconstruction**, **LeJEPA** (Joint-Embedding Predictive Architecture), and **SIGReg** (Isotropy Regularization) to prevent latent collapse.
*   **GAN Support:** Integrated **MPD/MSD discriminators** for adversarial waveform refinement.
*   **Optimization:** Added **ScaledAdam** and **Eden** schedulers for stable training of deep audio architectures.

### **Expected Outcome**
A functional research framework for high-fidelity audio reconstruction and robust latent representations, capable of performing well on downstream tasks (ASR, Emotion, Gender) while maintaining stability.

### **Hypothesis**
1.  **Zipformer + mHC:** Efficient transformer variants with manifold-constrained connections provide superior audio encoding.
2.  **SIGReg + LeJEPA:** Explicit isotropy regularization and joint-embedding prediction prevent latent collapse.
3.  **RAE-style Decoding:** Latent noise improves decoder robustness and audio generation quality.

---

## [fc65c71] tiihis commit should be signed
**Original Message:** tiihis commit should be signed

### **Changes Made**
*   Updated `README.md` with a single line: "This commit should be signed!".

### **Expected Outcome**
Verification of the git repository's configuration for GPG/SSH commit signing to ensure subsequent high-impact contributions are marked as "Verified."

### **Hypothesis**
The development environment's commit signing mechanism needed validation before proceeding with major architectural ports.

---

## [ac4c9b1] Fix train.py config mismatch and add 10% data training setup
**Original Message:** Fix train.py config mismatch and add 10% data training setup

### **Changes Made**
*   **Bug Fix:** Resolved a configuration mismatch where `SIGReg` loss weight was incorrectly passed as a structural parameter.
*   **Dataset Subsetting:** Added scripts to generate 10% and "smoke test" manifests for faster iteration.
*   **New Configs:** Introduced `train_10pct.yaml` (256-dim model) and `smoke_test_wandb.yaml`.
*   **Utility:** Added `count_params.py` and updated `.gitignore`.

### **Expected Outcome**
Ability to run medium-scale experiments to test model capacity and verify the `SIGReg` component within the PyTorch hierarchy.

### **Hypothesis**
A 10% data subset is sufficient to validate core architectural choices (mHC + JEPA + SIGReg) before full-scale training.

---

## [9f1ff48] changes
**Original Message:** changes

### **Changes Made**
*   **Decoder:** Replaced `nn.ConvTranspose1d` with `nn.Upsample` (linear) + `nn.Conv1d`.
*   **JEPA:** Removed `.detach()` from the "clean center" in loss calculation.
*   **GAN Training:** Integrated discriminator backward into `GradScaler` and added gradient clipping.
*   **Tooling:** Added NaN debugging, ASR/Classifier probing scripts, and reconstruction visualization.

### **Expected Outcome**
Reduction in "checkerboard" artifacts in audio, enhanced latent learning through bidirectional gradients in JEPA, and better training stability.

### **Hypothesis**
Removing the stop-gradient in JEPA leads to more informative representations, and upsampling + convolution produces higher-fidelity waveforms than transposed convolutions.

---

## [5c9035e] added enhancements
**Original Message:** added enhancements

### **Changes Made**
*   **LeJEPA:** Added masked and "mixture" (blended audio) training views.
*   **SIGReg:** Implemented formal statistical regularization using the **Epps-Pulley test**.
*   **GAN:** Integrated **Multi-Period (MPD)** and **Multi-Scale (MSD)** discriminators.
*   **Infrastructure:** Added `EXPERIMENT_GUIDE.md`, reproducibility scripts, and a unified research suite runner.

### **Expected Outcome**
Improved robustness to noise and overlap, non-collapsed latent space, and higher perceptual audio fidelity (eliminating "phasiness").

### **Hypothesis**
Invariance is learned through predictive modeling (JEPA), disentanglement through mixture recovery, and distributional stability through formal statistical tests (SIGReg).

---

## [ad3af18] added enhancements
**Original Message:** added enhancements

### **Changes Made**
*   **RAE Pivot**: Removed the tight 16-32 dim bottleneck; latent space $z$ now matches encoder $hE$ (256-dim).
*   **Noise Injection**: Enabled Gaussian noise injection during training with $\sigma_{max} = 0.1$.
*   **Documentation**: Added `ADR_001_RAE_Architecture.md`.

### **Expected Outcome**
Higher fidelity reconstruction and more "semantic" features for downstream tasks by avoiding "bottleneck strangulation."

### **Hypothesis**
Tight bottlenecks limit the effectiveness of LeJEPA; high-dimensional semantic latents better serve both reconstruction and representation learning.

---

## [48f8a48] changes
**Original Message:** changes

### **Changes Made**
*   **Variational Latents**: Added support for KL-divergence loss ($mu$ and $logvar$).
*   **Denoising Augmentations**: Integrated waveform-level noise and lowpass filtering.
*   **LeJEPA Refinement**: Added option to apply JEPA losses to intermediate high-dimensional features.
*   **Logging**: Enhanced WandB support for ASR tables and SIGReg variance stats.

### **Expected Outcome**
A more resilient model with a well-organized, isotropic latent manifold that avoids collapse.

### **Hypothesis**
VAE-style regularization combined with denoising objectives produces a more robust latent manifold.

---

## [cdaa578] new changes
**Original Message:** new changes

### **Changes Made**
*   **TorchDynamo Stability**: Moved stochastic whitening logic to `backward` pass to enable `torch.compile`.
*   **Capacity Change**: Reduced encoder `d_model` (192) but increased feedforward (768) and decoder channels (512).
*   **Training**: Enabled AMP, increased JEPA/STFT weights, and delayed GAN start to 10k steps.

### **Expected Outcome**
Significant boost in training throughput and improved latent robustness through `mix_recon` (reconstructing clean from mixed input).

### **Hypothesis**
Shifting randomness to the backward pass allows static graphs for compiler optimizations without losing regularization.

---

## [23dfb54] changes
**Original Message:** changes

### **Changes Made**
*   **Distributed Consistency**: Switched to explicit `dist.broadcast` for SIGReg random slices.
*   **Optimization**: Removed CPU-GPU synchronization points in `ScaledAdam`.
*   **Training Loop**: Switched to epoch-based iteration and added hardware acceleration flags (`tf32`).

### **Expected Outcome**
Elimination of distributed training deadlocks and increased iterations per second.

### **Hypothesis**
Explicit broadcasting is more reliable than seed-syncing for distributed manifold calculations; `.item()` calls were a major bottleneck.

---

## [cf04904] changes
**Original Message:** changes

### **Changes Made**
*   **Weight Rebalancing**: Reduced `stft_weight` and `jepa.weight` while increasing `sigreg.weight`.
*   **SIGReg Stability**: Reduced `t_max` to 1.0 for stronger gradients during distribution matching.
*   **Decoder Norm**: Enabled `latent_norm` to decouple scale from reconstruction.
*   **Memory**: Implemented CPU offloading for evaluation probes.

### **Expected Outcome**
Stable training process with a latent space strictly adhering to $N(0, 1)$ without SACRIFICING reconstruction quality.

### **Hypothesis**
A ~1:10 ratio between SIGReg and JEPA better enforces the isotropic Gaussian constraint; reducing bandwidth prevents vanishing gradients.

---

## [d33a29f] refactor and improvements
**Original Message:** refactor and improvements

### **Changes Made**
*   **SIGReg**: Increased `t_max` to 3.0 and added an explicit variance penalty.
*   **GAN Warmup**: Added a 5k-step linear warmup and reduced feature matching weight.
*   **Precision**: Introduced separate `GradScaler`s for G and D.
*   **Persistence**: Made latent statistics buffers persistent in the decoder.

### **Expected Outcome**
Finer-grained distribution matching and smoother GAN activation, leading to higher audio quality and checkpoint portability.

### **Hypothesis**
Explicit variance penalties act as a "safety net" for ECF-based matching; separate scalers prevent numerical cross-contamination.

---

## [12b4d30] added adaptive gan, fixed dead code
**Original Message:** added adaptive gan, fixed dead code

### **Changes Made**
*   **Adaptive GAN**: Implemented VQGAN-style dynamic loss weighting based on gradient ratios.
*   **Stability**: Reduced `d_lr` to `5.0e-5` and replaced blocking broadcasts with deterministic seeds.
*   **Cleanup**: Removed unused KL loss and `hE` JEPA logic.

### **Expected Outcome**
Self-correcting balance between reconstruction and adversarial training, reducing sensitivity to fixed hyperparameters.

### **Hypothesis**
Adaptive weighting provides a more robust balance than manual tuning, preventing the GAN from overpowering the primary objective.

---

## [88d58fc] fix gan loss, and eval probe
**Original Message:** fix gan loss, and eval probe

### **Changes Made**
*   **Precision Shift**: Forced GAN operations to `float32` outside of `autocast`.
*   **Adaptive Refinement**: Tightened adaptive weight clamp (10.0) and averaged losses by discriminator count.
*   **ASR Probe**: Increased segment length to 15.0s for better CTC convergence.

### **Expected Outcome**
Elimination of NaNs in adversarial training and more reliable WER metrics.

### **Hypothesis**
Adversarial training in this architecture is too numerically sensitive for `float16`; 15s is the "sweet spot" for temporal context in probes.

---

## [a035f79] optimizations for eval probe
**Original Message:** optimizations for eval probe

### **Changes Made**
*   **Memory Management**: ASR features stored on CPU; frozen encoder deleted after extraction.
*   **Performance**: Extraction now uses AMP; merged metadata extraction for emotion/gender.
*   **Resource Capping**: Added `max_samples` (2000) for ASR evaluation.

### **Expected Outcome**
Elimination of OOM crashes during training checkpoints and reduced "dead time" during evaluations.

### **Hypothesis**
Moving large feature tensors to CPU and using AMP for the encoder maintains evaluation quality while drastically reducing hardware requirements.

---

## [70913b4] optimizations for eval probe
**Original Message:** optimizations for eval probe

### **Changes Made**
*   **Adaptive Weight Refinement**: Forced `gan_w` to be a Python scalar via `.item()`.

### **Expected Outcome**
Predictable scalar behavior in the loss backward pass, avoiding potential graph retention issues.

### **Hypothesis**
Ensuring the weight is a constant scalar improves stability and maintains graph cleanliness.

---

## [85ec22b] optimizations for eval probe
**Original Message:** optimizations for eval probe

### **Changes Made**
*   **Memory Allocator**: Enabled `expandable_segments:True` via `PYTORCH_CUDA_ALLOC_CONF`.
*   **AMP for GAN**: Moved GAN forward passes into `autocast` (float16) but kept loss calculation in `float32`.
*   **Scaling**: Introduced a dedicated `d_scaler` for the discriminator.

### **Expected Outcome**
Significant VRAM savings without numerical divergence, allowing heavier probes to run reliably.

### **Hypothesis**
AMP activations are stable as long as the loss and updates remain in `float32`; `expandable_segments` reduces fragmentation OOMs.

---

## [b750e7e] optimizations
**Original Message:** optimizations

### **Changes Made**
*   **Denoising Objective**: Decoder now reconstructs from **masked/augmented** latents.
*   **Zero-Mean Reg**: Added a penalty for non-zero latent means to SIGReg.
*   **MHC Scaling**: Added learned `branch_scale` (init 0) to MHC layers.
*   **Rank Monitoring**: Integrated Participation Ratio (`z_rank`) logging.

### **Expected Outcome**
More robust, global features through MAE-style reconstruction and standardized latent distributions.

### **Hypothesis**
Reconstructing from masked latents provides a stronger training signal; initializing MHC branches to zero allows gradual complexity integration.

---

## [51123b7] optimizations
**Original Message:** optimizations

### **Changes Made**
*   **Masked Audio Restoration (MAR)**: Reconstruction losses now only penalize the masked-out regions.
*   **SIGReg Simplification**: Removed the mean penalty, focusing solely on variance control.

### **Expected Outcome**
Encoder is forced to "inpaint" missing signal using global context, leading to richer semantic representations.

### **Hypothesis**
Inpainting is a harder and more meaningful pretext task than full reconstruction; variance control is the primary factor for stability.

---

## [c215266] optimizations
**Original Message:** optimizations

### **Changes Made**
*   **Spectral Masking**: Moved masking from time-domain to frequency-domain (STFT magnitudes).
*   **Loss Normalization**: Normalized STFT loss by the mask fraction.
*   **Weight Boost**: Significantly increased JEPA and SIGReg weights.

### **Expected Outcome**
Cleaner reconstruction signal by eliminating time-domain boundary artifacts; stronger enforcement of latent constraints.

### **Hypothesis**
Masking spectrogram magnitudes is more numerically stable and semantically relevant than raw waveform masking.

---

## [8d17f8b] optimizations
**Original Message:** optimizations

### **Changes Made**
*   **InfoNCE Transition**: Replaced cosine-similarity JEPA with **InfoNCE contrastive loss**.
*   **SIGReg Decomposition**: Split latent space into utterance-level means and frame-level residuals for separate testing.
*   **STFT Fix**: Stabilized Spectral Convergence denominator for masked regions.
*   **Decoder Target**: Reverted decoder to reconstruct from **clean** latents while the encoder handles masking via InfoNCE.

### **Expected Outcome**
Steady decrease in primary loss (avoiding stagnation) and more granular control over latent structure.

### **Hypothesis**
Explicit negative sampling (InfoNCE) is required to prevent collapse; decoupling decoder from the masking task stabilizes reconstruction.

---

## [6401535] optimizations
**Original Message:** optimizations

### **Changes Made**
*   **Numerical Robustness**: Replaced hardcoded `-1e9` in InfoNCE with `finfo.min`.
*   **Stability**: Removed `torch.compile` to eliminate startup latency and runtime instability.

### **Expected Outcome**
Mathematically robust mask for all floating-point types and more predictable training behavior.

### **Hypothesis**
`torch.compile` overhead and potential black-box bugs outweighed the speed benefits for the current architecture.

---

## [bd85807] optimizations
**Original Message:** optimizations

### **Changes Made**
*   **Hierarchical Rank**: Added metrics for diversity between samples (`z_rank_utt`) and within samples (`z_rank_res`).
*   **RMS Tracking**: Added latent magnitude and prediction error RMS logs.
*   **Collapse Ratio**: Integrated `jepa_to_norm_ratio`.

### **Expected Outcome**
Empirical verification of representation health, distinguishing between genuine learning and zero-collapse failure modes.

### **Hypothesis**
JEPA loss is ambiguous without tracking latent magnitudes; hierarchical rank analysis detects dimensional collapse.

---

## [d42713b] optimizations
**Original Message:** optimizations

### **Changes Made**
*   **Deterministic Augs**: Increased noise/lowpass/gain probabilities to **1.0**.
*   **Per-Sample Masking**: Refactored augmentation to apply unique masks to each batch element.

### **Expected Outcome**
Superior generalization to "in-the-wild" audio through constant acoustic distortion and reduced batch correlation.

### **Hypothesis**
Forcing the model to always handle distortions develops natural invariance; per-sample masking prevents reliance on batch-level regularities.
