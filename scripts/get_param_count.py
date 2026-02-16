import torch
from models.frontend_conv import ConvFrontend, FrontendConfig
from models.encoder import Encoder, EncoderConfig
from models.decoder_generator import WaveformDecoder, DecoderConfig
from utils.config import load_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    cfg = load_config("configs/exp0.yaml")
    mcfg = cfg["model"]
    
    frontend = ConvFrontend(FrontendConfig(**mcfg["frontend"]))
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg["encoder"]))
    
    latent_dim = int(mcfg["encoder"]["d_model"])
    decoder = WaveformDecoder(latent_dim, DecoderConfig(**mcfg["decoder"]))
    
    f_params = count_parameters(frontend)
    e_params = count_parameters(encoder)
    d_params = count_parameters(decoder)
    
    print(f"Frontend: {f_params:,}")
    print(f"Encoder:  {e_params:,}")
    print(f"Decoder:  {d_params:,}")
    print(f"Total:    {f_params + e_params + d_params:,}")

if __name__ == "__main__":
    main()
