# Architecture and Training Explanation

## 1. Architecture

The system is a **Continuous Latent Autoencoder** designed for audio representation learning. It consists of a hierarchical encoder-decoder structure with a regularization bottleneck.

### **Frontend** (`models/frontend_conv.py`)
- **Input**: Raw waveform audio `(Batch, 1, Time)`.
- **Mechanism**: A stack of Strided 1D Convolutions followed by GroupNorm and GELU activations.
- **Purpose**: downsamples the high-frequency waveform into a sequence of low-level feature tokens `(Batch, Channels, Time')`.

### **Encoder** (`models/encoder.py`)
- **Input**: Feature tokens from the frontend.
- **Mechanism**:
  - **Zipformer2 Blocks**: Efficient transformer-like layers (from `models/zipformer.py`) for processing sequences.
  - **MHC (Manifold-Constrained Hyper-Connections)**: Optional mechanism to improve information flow and representation learning (`models/mhc.py`).
  - **Positional Encoding**: Compact Relative Positional Encoding.
- **Purpose**: Transforms low-level features into a rich, contextualized latent representation.

### **Bottleneck** (`models/encoder.py`)
- **Mechanism**: A 1D Convolution (`kernel_size=1`) projecting to the latent dimension, followed by normalization (`LayerNorm` or `RMSNorm`).
- **Purpose**: Compresses the representation into a deterministic latent space `z` `(Batch, Latent_Dim, Time')`.

### **Decoder** (`models/decoder_generator.py`)
- **Input**: Latent representation `z`.
- **Mechanism**:
  - **Upsampling**: A sequence of Transposed 1D Convolutions to reach the original waveform resolution.
  - **FiLM Conditioning**: `ResBlockFiLM` layers modulate the features based on the latent `z` (Feature-wise Linear Modulation).
- **Purpose**: Reconstructs the original waveform from the compressed latent representation.

---

## 2. Loss Function

The model is trained using a composite loss function that encourages reconstruction fidelity, latent space regularity, and predictive robustness.

### **Reconstruction Losses**
- **Multi-Resolution STFT Loss**: Measures the spectral difference between the original and reconstructed waveforms across multiple time-frequency resolutions. This is the primary driver for audio quality.
- **L1 Waveform Loss**: Direct L1 distance between the waveforms `|x - x_hat|`.

### **LeJEPA (Joint Embedding Predictive Architecture) Loss**
- **Goal**: Encourages the model to learn robust features that are invariant to noise and masking.
- **Mechanism**:
  1. **Clean View**: Encode the original audio to get target latent `z_clean`.
  2. **Masked View**: Apply masking to the frontend features of the same audio, then encode to get `z_masked`.
  3. **Loss**: MSE between `z_clean` and `z_masked`.
- **Mix View (Exp1+)**: Optionally mixes two audio samples and forces the encoder to recover the "primary" source's latent from the mix.

### **SIGReg (Signal Regularization) Loss**
- **Goal**: Prevents collapse and encourages information content in the latent space.
- **Mechanism**: Computes statistics on the latent embeddings to ensure variance and decorrelation (implicitly maximizing entropy or similar information-theoretic properties).

### **GAN Losses (Optional)**
- If enabled, Multi-Period (MPD) and Multi-Scale (MSD) discriminators are used.
- **Adversarial Loss**: Generator tries to fool discriminators.
- **Feature Matching Loss**: Generator minimizes the difference between discriminator feature maps of real vs. fake audio.

---

## 3. Training Process

Training is managed by `train.py` and follows a standard PyTorch loop with specific augmentations.

1.  **Initialization**: Config loading, seed setting, and model/optimizer instantiation (ScaledAdam or AdamW).
2.  **Forward Pass**:
    - **Clean Path**: Encode `wav` $\to$ `z`. Decode `z` $\to$ `wav_recon`.
    - **Masked Path**: Mask frontend features $\to$ Encode $\to$ `z_masked`.
    - **Loss Calculation**: Compute STFT loss on `wav_recon`, LeJEPA loss between `z` and `z_masked`, and SIGReg loss on latents.
3.  **Backward Pass**: Gradients are accumulated (`grad_accum_steps`) and scaled (AMP).
4.  **Optimization**: Optimizer step and learning rate scheduling (Eden or Eden2).
5.  **Logging & Validation**: Stats are logged to `jsonl`/WandB. Periodic validation evaluates reconstruction and latent quality on a held-out set.
6.  **Checkpointing**: Models are saved periodically. Optional "probes" (ASR, Emotion, Gender classification) can run on checkpoints to evaluate downstream utility.

---

## 4. Data Loading

Data is handled via a manifest-based system in `data/dataset.py`.

- **Manifest**: A JSONL file where each line contains metadata (filepath, duration, etc.) for an audio file.
- **Dataset (`AudioManifestDataset`)**:
  - Reads the manifest.
  - Loads audio on-the-fly.
  - Chunks audio into fixed-length segments (defined by `segment_seconds` in config).
- **DataLoader**:
  - Batches samples using `collate_fixed`.
  - Supports multi-process loading (`num_workers`).
- **Augmentation**:
  - **Feature Masking**: Applied in the training loop (not the dataloader) to frontend features.
  - **Mixing**: A secondary iterator allows mixing two different samples for the "Mix View" training objective.

---

## 5. Technical Implementation Details

### Distributed Training
**Status: Not Currently Supported**
While the codebase contains some primitives for distributed computing, the main training entrypoint (`train.py`) is currently designed for **Single-GPU** execution.
- It does not initialize `torch.distributed`.
- It lacks `DistributedDataParallel` wrappers.
- To run on multiple GPUs, `train.py` would need to be modified to handle process groups and distributed sampling.

### LeJEPA & SIGReg Fidelity
The implementation in `models/sigreg.py` closely follows the **LeJEPA** paper ("Stable and Scalable Implementation"):
- **SIGReg**: Implements the **Epps-Pulley** test using trapezoidal quadrature (default 17 knots) and random slicing (`SlicingUnivariateTest`), matching Algorithm 1 of the reference method.
- **Distributed Awareness**: The SIGReg module includes `_all_reduce` logic to synchronize statistics across GPUs, ensuring the "global" distribution is measured correctly if DDP were enabled.
- **Prediction Loss**: `train.py` minimizes the L2 distance between clean and masked embeddings, equivalent to the prediction loss formulation in LeJEPA.
