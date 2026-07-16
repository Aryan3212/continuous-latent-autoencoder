from __future__ import annotations

import torch.nn as nn

from models.decoder_generator import WaveformDecoder
from models.encoder import Encoder
from models.frontend_conv import ConvFrontend
from models.projector import Projector
from schema import ModelCfg


class Autoencoder(nn.Module):
    """Frontend, encoder, projector, and waveform decoder."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.frontend = ConvFrontend(cfg.frontend)
        self.encoder = Encoder(self.frontend.out_channels, cfg.encoder)
        self.projector = Projector(cfg.encoder.d_model, cfg.projector)
        self.decoder = WaveformDecoder(cfg.encoder.d_model, cfg.decoder)
