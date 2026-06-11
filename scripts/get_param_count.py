import torch
from models.frontend_conv import ConvFrontend
from models.encoder import Encoder
from models.decoder_generator import WaveformDecoder
from models.projector import Projector
from utils.config import load_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    cfg = load_config("configs/exp0.yaml")

    frontend = ConvFrontend(cfg.model.frontend)
    encoder = Encoder(frontend.out_channels, cfg.model.encoder)
    latent_dim = cfg.model.encoder.d_model
    decoder = WaveformDecoder(latent_dim, cfg.model.decoder)
    projector = Projector(latent_dim, cfg.model.projector)

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
