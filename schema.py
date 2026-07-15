from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    """Forbid unknown keys so typos get caught at startup."""
    model_config = ConfigDict(extra="forbid")


class WandbCfg(_Base):
    enabled: bool = True
    project: str = "continuous-latent-ae"
    name: Optional[str] = None


class RunCfg(_Base):
    run_id: Optional[str] = None
    out_dir: str = "runs"
    seed: int = 0
    amp: bool = True
    # Precision used by the AMP autocast region. "fp16" preserves the old default
    # (good exponent range is NOT guaranteed); "bf16" is recommended on Ampere+
    # hardware (4090 / A100): it keeps an FP32-like exponent range through the
    # dynamically ranged parts of the network while preserving AMP throughput.
    # The STFT / complex path is always forced to FP32 regardless of this setting.
    amp_dtype: str = "fp16"
    # Per-process VRAM cap for PyTorch's allocator. Note the CUDA context + NCCL
    # (~0.9 GiB) live OUTSIDE this cap, so fraction*total + that must stay under
    # the card; back off if you OOM right at the cap.
    gpu_mem_fraction: float = Field(0.92, gt=0.0, le=1.0)
    wandb: WandbCfg = Field(default_factory=WandbCfg)


class DataCfg(_Base):
    sample_rate: int = 16000
    segment_seconds: float = 3.0
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2  # batches each worker buffers ahead; raise to hide decode/resample spikes


class WaveAugCfg(_Base):
    enabled: bool = False
    noise_prob: float = 0.0
    noise_snr_min: float = 3.0
    noise_snr_max: float = 20.0
    lowpass_prob: float = 0.0
    lowpass_min_freq: float = 2000.0
    lowpass_max_freq: float = 8000.0
    gain_prob: float = 0.0
    gain_min: float = 0.5
    gain_max: float = 1.5
    clip_prob: float = 0.0
    clip_min: float = 0.5


class WaveChunkMaskCfg(_Base):
    enabled: bool = True
    target_ratio: float = 0.25
    min_span_frames: int = 2
    max_span_frames: int = 8


class AugCfg(_Base):
    wave_aug: WaveAugCfg = Field(default_factory=WaveAugCfg)
    wave_chunk_mask: WaveChunkMaskCfg = Field(default_factory=WaveChunkMaskCfg)


class FrontendCfg(_Base):
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    groups: int = 1


class MHCCfg(_Base):
    enabled: bool = True
    num_streams: int = 2
    start_layer: int = 2
    period: int = 3
    sinkhorn_iters: int = 10
    tau: float = 0.05
    dropout: float = 0.0
    identity_mix: bool = True
    alpha_init: float = 0.01


class EncoderCfg(_Base):
    # Selects the per-layer block. "conformer" = macaron Conformer;
    # "fastconformer" = same block + Squeeze-and-Excitation (and typically a
    # smaller cnn_module_kernel, e.g. 9). Both share d_model/n_layers/heads/ffn.
    encoder_type: str = "conformer"   # "conformer" | "fastconformer"
    d_model: int = 256
    n_layers: int = 4
    num_heads: int = 4
    feedforward_dim: int = 768
    dropout: float = 0.1
    cnn_module_kernel: int = 31
    use_se: bool = True                # FastConformer SE gate (only used when encoder_type=="fastconformer")
    xscaling: bool = False             # scale input embeddings by sqrt(d_model) (NeMo FastConformer)
    mhc: MHCCfg = Field(default_factory=MHCCfg)


class DecoderCfg(_Base):
    channels: int = 256
    up_strides: List[int] = Field(default_factory=lambda: [4, 4, 4, 4, 5])
    up_kernels: List[int] = Field(default_factory=lambda: [8, 8, 8, 8, 10])
    res_blocks_per_up: int = 2
    res_dilations: List[int] = Field(default_factory=lambda: [1, 3, 9])
    film_hidden: int = 128


class ProjectorCfg(_Base):
    hidden_dim: int = 2048
    output_dim: int = 64
    n_hidden_layers: int = 2


class ModelCfg(_Base):
    frontend: FrontendCfg
    encoder: EncoderCfg = Field(default_factory=EncoderCfg)
    decoder: DecoderCfg = Field(default_factory=DecoderCfg)
    projector: ProjectorCfg = Field(default_factory=ProjectorCfg)


class STFTCfg(_Base):
    fft_sizes: List[int] = Field(default_factory=lambda: [256, 512, 1024, 2048])
    hop_ratio: float = 0.25
    win_ratio: float = 1.0
    center: bool = True
    window: str = "hann"
    logmag_eps: float = 1.0e-3
    # Speech recon is carried mostly by magnitude / log-magnitude terms; spectral
    # convergence (sc) is deliberately down-weighted so it doesn't dominate.
    sc_weight: float = 0.1
    mag_weight: float = 1.0
    logmag_weight: float = 1.0


class MelCfg(_Base):
    """Mel-spectrogram reconstruction loss config.

    The mel loss operates on mel-scaled magnitudes, which are inherently
    magnitude / log-magnitude quantities — so it is weighted toward mag / log_mag
    by construction (spectral convergence is off by default via sc_weight=0.0).
    Shares the sc/mag/logmag weighting scheme with STFTCfg for a clean ablation.
    """
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    sample_rate: int = 16000
    fmin: float = 0.0
    fmax: Optional[float] = None       # None -> sample_rate / 2
    window: str = "hann"
    logmag_eps: float = 1.0e-3
    sc_weight: float = 0.0            # mel is log-mag; SC off by default
    mag_weight: float = 1.0
    logmag_weight: float = 1.0


class JEPACfg(_Base):
    weight: float = 1.0
    num_globals: int = 2
    num_locals: int = 4


class SIGRegCfg(_Base):
    weight: float = 0.05
    num_slices: int = 1024
    t_max: float = 5.0
    n_points: int = 17


class VISRegCfg(_Base):
    """VISReg (Vector-ISotropic Gaussianisation), https://haiyuwu.github.io/visreg/.

    Frame-level Gaussianisation on the projector output, mirroring SIGReg's role
    (the two are mutually exclusive; select via ``LossCfg.reg_type``). No
    learnable params; the random projection is resampled each forward pass.
    """

    weight: float = 0.05
    num_projections: int = 256


class AdvCfg(_Base):
    """HiFi-GAN-style adversarial + feature-matching loss on the decoder output.

    Disabled by default so existing configs/runs are unaffected. When enabled,
    train.py builds a Multi-Period Discriminator and a second optimizer.
    """
    enabled: bool = False
    adv_weight: float = 1.0
    fm_weight: float = 2.0
    adv_start_step: int = 0          # generator adversarial term active from here
    fm_start_step: int = 20000       # feature-matching term active from here
    lr: float = 2.0e-4               # discriminator AdamW lr
    betas: List[float] = Field(default_factory=lambda: [0.8, 0.99])
    periods: List[int] = Field(default_factory=lambda: [2, 3, 5, 7, 11])
    # MPD hidden channel widths. HiFi-GAN default (32,128,512,1024) is a ~41M
    # discriminator — too heavy for 6 GB; slim default keeps it ~proportionate
    # to the generator. Raise on a bigger GPU.
    disc_channels: List[int] = Field(default_factory=lambda: [16, 64, 128, 256])
    loss_type: str = "lsgan"         # "lsgan" (least-squares) | "hinge" (margin-based).
                                      # Pick ONE and keep it for the whole run; do not combine.
                                      # Hinge is preferred for the multi-discriminator baseline:
                                      # it does not square growing logits and gives a linear
                                      # generator objective.
    # Adaptive adversarial weight (VQGAN-style). When on, l_adv is rescaled each
    # step so its gradient at the decoder's last layer matches the reconstruction
    # gradient there, then multiplied by adv_weight. This keeps the GAN from
    # overwhelming reconstruction regardless of the raw l_adv magnitude. With it
    # on, adv_weight becomes a relative-strength knob (1.0 = parity with recon).
    adaptive: bool = False
    adaptive_max: float = 10      # clamp on the adaptive lambda (VQGAN default)


class LossCfg(_Base):
    # Reconstruction loss: swap between "stft" and "mel" for the ablation.
    recon_type: str = "stft"          # "stft" | "mel" — which recon loss is trained
    recon_weight: float = 0.005       # weight applied to whichever recon loss is active
    recon_log_start_step: int = 1000  # only log the STFT log-mag metric after this many steps
    # Gaussianisation loss: swap between "sigreg" and "visreg" for the ablation.
    # Both act frame-level on the projector output (gathered across ranks so the
    # estimate uses the global batch). Whichever is active is logged under
    # ``l_sig`` / ``l_vis`` respectively.
    reg_type: str = "sigreg"          # "sigreg" | "visreg" — which Gaussianisation loss is trained
    stft: STFTCfg = Field(default_factory=STFTCfg)
    mel: MelCfg = Field(default_factory=MelCfg)
    jepa: JEPACfg = Field(default_factory=JEPACfg)
    sigreg: SIGRegCfg = Field(default_factory=SIGRegCfg)
    visreg: VISRegCfg = Field(default_factory=VISRegCfg)
    adv: AdvCfg = Field(default_factory=AdvCfg)


class SchedulerCfg(_Base):
    warmup_steps: int = 2000
    # None -> derived from train.max_steps at the Config level (they are always
    # equal in practice). Set explicitly only to decay over a different horizon.
    total_steps: Optional[int] = None
    min_lr_ratio: float = 0.0


class OptimCfg(_Base):
    lr: float = 5.0e-4
    betas: List[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8
    weight_decay: float = 1.0e-2
    scheduler: SchedulerCfg = Field(default_factory=SchedulerCfg)
    grad_clip: float = 1.0


class TrainCfg(_Base):
    batch_size: int = 50
    grad_accum_steps: int = 1
    max_steps: int = 30000
    log_interval_steps: int = 10
    eval_interval_steps: int = 5000
    save_interval_steps: int = 10000
    probe_interval_steps: int = 1000  # embedding similarity probe (pos/neg MSE on z)
    val_batches: Optional[int] = None


class EmotionCfg(_Base):
    enabled: bool = False
    train_manifest: Optional[str] = None
    dev_manifest: Optional[str] = None
    label_key: str = "emotion"
    steps: int = 2000
    batch_size: int = 64
    segment_seconds: Optional[float] = None


class GenderCfg(_Base):
    enabled: bool = False
    train_manifest: Optional[str] = None
    dev_manifest: Optional[str] = None
    label_key: str = "gender"
    steps: int = 1500
    batch_size: int = 64
    segment_seconds: Optional[float] = None


class AsrCfg(_Base):
    enabled: bool = True
    train_manifest: Optional[str] = None
    dev_manifest: Optional[str] = None
    text_key: str = "text"
    steps: int = 1000
    batch_size: int = 16
    segment_seconds: float = 15.0
    max_samples: int = 500


class EvalCfg(_Base):
    enabled: bool = True
    emotion: EmotionCfg = Field(default_factory=EmotionCfg)
    gender: GenderCfg = Field(default_factory=GenderCfg)
    asr: AsrCfg = Field(default_factory=AsrCfg)


class Config(_Base):
    resolved_config_path: Optional[str] = None
    run: RunCfg = Field(default_factory=RunCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    aug: AugCfg = Field(default_factory=AugCfg)
    model: ModelCfg
    loss: LossCfg = Field(default_factory=LossCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
