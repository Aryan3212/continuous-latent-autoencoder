from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WandbCfg(_Base):
    enabled: bool = True
    project: str = "continuous-latent-ae"
    name: str | None = None


class RunCfg(_Base):
    run_id: str | None = None
    out_dir: str = "runs"
    seed: int = 0
    amp: bool = True
    amp_dtype: Literal["fp16", "bf16"] = "fp16"
    gpu_mem_fraction: float = Field(0.92, gt=0.0, le=1.0)
    wandb: WandbCfg = Field(default_factory=WandbCfg)


class DataCfg(_Base):
    sample_rate: int = 16000
    segment_seconds: float = 3.0
    train_manifest: str
    val_manifest: str | None = None
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2


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


class SpanMaskCfg(_Base):
    enabled: bool = False
    ratio: float = Field(0.25, ge=0.0, le=1.0)
    min_span_frames: int = Field(2, ge=1)
    max_span_frames: int = Field(8, ge=1)

    @model_validator(mode="after")
    def validate_span_range(self) -> "SpanMaskCfg":
        if self.max_span_frames < self.min_span_frames:
            raise ValueError("max_span_frames must be >= min_span_frames")
        return self


class NoiseCfg(_Base):
    enabled: bool = False
    std: float = Field(0.0, ge=0.0)


class AugCfg(_Base):
    waveform_aug_global: WaveAugCfg = Field(default_factory=WaveAugCfg)
    waveform_aug_local: WaveAugCfg | None = None
    waveform_aug_local_mask: SpanMaskCfg = Field(default_factory=SpanMaskCfg)
    frontend_frame_local_mask: SpanMaskCfg = Field(default_factory=SpanMaskCfg)
    frontend_frame_noise: NoiseCfg = Field(default_factory=NoiseCfg)
    decoder_input_mask: SpanMaskCfg = Field(default_factory=SpanMaskCfg)
    decoder_input_noise: NoiseCfg = Field(default_factory=NoiseCfg)


class FrontendCfg(_Base):
    channels: list[PositiveInt]
    kernels: list[PositiveInt]
    strides: list[PositiveInt]
    groups: int = Field(1, ge=1)

    @model_validator(mode="after")
    def validate_stack(self) -> "FrontendCfg":
        if not self.channels or not (
            len(self.channels) == len(self.kernels) == len(self.strides)
        ):
            raise ValueError("channels, kernels, and strides must have the same non-zero length")
        return self


class MHCCfg(_Base):
    enabled: bool = True
    num_streams: int = Field(2, ge=1)
    start_layer: int = Field(2, ge=0)
    period: int = Field(3, ge=1)
    sinkhorn_iters: int = Field(10, ge=1)
    tau: float = Field(0.05, gt=0.0)
    dropout: float = Field(0.0, ge=0.0, le=1.0)
    identity_mix: bool = True
    alpha_init: float = 0.01

    @model_validator(mode="after")
    def validate_identity_mix(self) -> "MHCCfg":
        if self.identity_mix and not 0.0 < self.alpha_init < 1.0:
            raise ValueError("alpha_init must be in (0, 1) when identity_mix is enabled")
        return self


class EncoderCfg(_Base):
    encoder_type: Literal["conformer", "fastconformer"] = "conformer"
    d_model: int = Field(256, ge=1)
    n_layers: int = Field(4, ge=1)
    num_heads: int = Field(4, ge=1)
    feedforward_dim: int = Field(768, ge=1)
    dropout: float = 0.1
    cnn_module_kernel: PositiveInt = 31
    use_se: bool = True
    xscaling: bool = False
    mhc: MHCCfg = Field(default_factory=MHCCfg)

    @model_validator(mode="after")
    def validate_attention(self) -> "EncoderCfg":
        if self.d_model % (2 * self.num_heads):
            raise ValueError("d_model must be divisible by 2 * num_heads")
        if self.cnn_module_kernel % 2 == 0:
            raise ValueError("cnn_module_kernel must be odd")
        return self


class DecoderCfg(_Base):
    channels: int = 256
    up_strides: list[PositiveInt] = Field(default_factory=lambda: [4, 4, 4, 4, 5])
    up_kernels: list[PositiveInt] = Field(default_factory=lambda: [8, 8, 8, 8, 10])
    res_blocks_per_up: int = 2
    res_dilations: list[int] = Field(default_factory=lambda: [1, 3, 9])
    film_hidden: int = 128

    @model_validator(mode="after")
    def validate_stack(self) -> "DecoderCfg":
        if not self.up_strides or len(self.up_strides) != len(self.up_kernels):
            raise ValueError("up_strides and up_kernels must have the same non-zero length")
        return self


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
    fft_sizes: list[int] = Field(default_factory=lambda: [256, 512, 1024, 2048])
    hop_ratio: float = 0.25
    win_ratio: float = 1.0
    center: bool = True
    window: str = "hann"
    logmag_eps: float = 1.0e-3
    sc_weight: float = 0.1
    mag_weight: float = 1.0
    logmag_weight: float = 1.0


class MelCfg(_Base):
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    fmin: float = 0.0
    fmax: float | None = None
    window: str = "hann"
    logmag_eps: float = 1.0e-3
    sc_weight: float = 0.0
    mag_weight: float = 1.0
    logmag_weight: float = 1.0


class JEPACfg(_Base):
    weight: float = 1.0
    num_globals: int = Field(2, ge=1)
    num_locals: int = Field(4, ge=1)


class SIGRegCfg(_Base):
    weight: float = 0.05
    num_slices: int = Field(1024, ge=1)
    t_max: float = 5.0
    n_points: int = Field(17, ge=3)

    @model_validator(mode="after")
    def validate_points(self) -> "SIGRegCfg":
        if self.n_points % 2 == 0:
            raise ValueError("n_points must be odd")
        return self


class VISRegCfg(_Base):
    weight: float = 0.05
    num_projections: int = Field(256, ge=1)


class AdvCfg(_Base):
    enabled: bool = False
    adv_weight: float = 1.0
    fm_weight: float = 2.0
    adv_start_step: int = 0
    fm_start_step: int = 20000
    lr: float = 2.0e-4
    betas: list[float] = Field(default_factory=lambda: [0.8, 0.99])
    periods: list[int] = Field(default_factory=lambda: [2, 3, 5, 7, 11])
    disc_channels: list[int] = Field(default_factory=lambda: [16, 64, 128, 256])
    loss_type: Literal["lsgan", "hinge"] = "lsgan"
    adaptive: bool = False
    adaptive_max: float = 10


class LossCfg(_Base):
    recon_type: Literal["stft", "mel"] = "stft"
    recon_views: Literal["global", "local", "all"] = "global"
    recon_weight: float = 0.005
    recon_log_start_step: int = 1000
    reg_type: Literal["sigreg", "visreg"] = "sigreg"
    stft: STFTCfg = Field(default_factory=STFTCfg)
    mel: MelCfg = Field(default_factory=MelCfg)
    jepa: JEPACfg = Field(default_factory=JEPACfg)
    sigreg: SIGRegCfg = Field(default_factory=SIGRegCfg)
    visreg: VISRegCfg = Field(default_factory=VISRegCfg)
    adv: AdvCfg = Field(default_factory=AdvCfg)


class SchedulerCfg(_Base):
    warmup_steps: int = Field(2000, ge=0)
    total_steps: int | None = Field(None, ge=1)
    min_lr_ratio: float = Field(0.0, ge=0.0, le=1.0)


class OptimCfg(_Base):
    lr: float = 5.0e-4
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8
    weight_decay: float = 1.0e-2
    scheduler: SchedulerCfg = Field(default_factory=SchedulerCfg)
    grad_clip: float = 1.0


class TrainCfg(_Base):
    batch_size: int = Field(50, ge=1)
    grad_accum_steps: int = Field(1, ge=1)
    max_steps: int = Field(30000, ge=1)
    log_interval_steps: int = Field(10, ge=1)
    eval_interval_steps: int = Field(5000, ge=1)
    save_interval_steps: int = Field(10000, ge=1)
    probe_interval_steps: int = Field(1000, ge=1)
    val_batches: int | None = None


class EmotionCfg(_Base):
    enabled: bool = False
    train_manifest: str | None = None
    dev_manifest: str | None = None
    label_key: str = "emotion"
    steps: int = 2000
    batch_size: int = 64
    segment_seconds: float | None = None


class GenderCfg(_Base):
    enabled: bool = False
    train_manifest: str | None = None
    dev_manifest: str | None = None
    label_key: str = "gender"
    steps: int = 1500
    batch_size: int = 64
    segment_seconds: float | None = None


class AsrCfg(_Base):
    enabled: bool = True
    train_manifest: str | None = None
    dev_manifest: str | None = None
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
    run: RunCfg = Field(default_factory=RunCfg)
    data: DataCfg
    aug: AugCfg = Field(default_factory=AugCfg)
    model: ModelCfg
    loss: LossCfg = Field(default_factory=LossCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)

    @model_validator(mode="after")
    def validate_frame_rate(self) -> "Config":
        if math.prod(self.model.frontend.strides) != math.prod(self.model.decoder.up_strides):
            raise ValueError("frontend and decoder stride products must match")
        return self
