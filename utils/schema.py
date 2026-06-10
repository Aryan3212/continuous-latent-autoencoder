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
    device: str = "auto"
    amp: bool = True
    wandb: WandbCfg = Field(default_factory=WandbCfg)


class DataCfg(_Base):
    sample_rate: int = 16000
    segment_seconds: float = 3.0
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False


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
    d_model: int = 256
    n_layers: int = 4
    num_heads: int = 4
    feedforward_dim: int = 768
    dropout: float = 0.1
    cnn_module_kernel: int = 31
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
    sc_weight: float = 1.0
    mag_weight: float = 1.0
    logmag_weight: float = 1.0


class JEPACfg(_Base):
    weight: float = 1.0
    num_globals: int = 2
    num_locals: int = 4
    context_weight: float = 1.0


class SIGRegCfg(_Base):
    weight: float = 0.05
    z_weight: float = 0.0      # frame-level SIGReg on encoder z (0 = off)
    utt_weight: float = 0.0    # utterance-level SIGReg on pooled p (0 = off)
    num_slices: int = 1024
    t_max: float = 5.0
    n_points: int = 17


class LossCfg(_Base):
    stft_weight: float = 0.005
    stft: STFTCfg = Field(default_factory=STFTCfg)
    wav_l1_weight: float = 0.0
    jepa: JEPACfg = Field(default_factory=JEPACfg)
    sigreg: SIGRegCfg = Field(default_factory=SIGRegCfg)


class SchedulerCfg(_Base):
    warmup_steps: int = 2000
    total_steps: int = 200000
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
    use_latent: bool = False


class EvalCfg(_Base):
    enabled: bool = True
    emotion: EmotionCfg = Field(default_factory=EmotionCfg)
    gender: GenderCfg = Field(default_factory=GenderCfg)
    asr: AsrCfg = Field(default_factory=AsrCfg)


class Config(_Base):
    model_config = ConfigDict(extra="allow")

    resolved_config_path: Optional[str] = None
    run: RunCfg = Field(default_factory=RunCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    aug: AugCfg = Field(default_factory=AugCfg)
    model: ModelCfg
    loss: LossCfg = Field(default_factory=LossCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
