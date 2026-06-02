import torch
from models.frontend_conv import ConvFrontend, FrontendConfig
from models.encoder import Encoder, EncoderConfig
from models.decoder_generator import WaveformDecoder, DecoderConfig
from models.projector import Projector, ProjectorConfig
from utils.config import load_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    cfg = load_config("configs/exp0.yaml")

    frontend = ConvFrontend(FrontendConfig(**cfg.model.frontend.model_dump()))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**cfg.model.encoder.model_dump()))
    latent_dim = cfg.model.encoder.d_model
    decoder = WaveformDecoder(latent_dim, DecoderConfig(**cfg.model.decoder.model_dump()))
    projector = Projector(latent_dim, ProjectorConfig(**cfg.model.projector.model_dump()))

    f_params = count_parameters(frontend)
    e_params = count_parameters(encoder)
    d_params = count_parameters(decoder)
    p_params = count_parameters(projector)

    print(f"Frontend:  {f_params:>10,}")
    print(f"Encoder:   {e_params:>10,}")
    print(f"Decoder:   {d_params:>10,}")
    print(f"Projector: {p_params:>10,}")
    print(f"Total:     {f_params + e_params + d_params + p_params:>10,}")

if __name__ == "__main__":
    main()
