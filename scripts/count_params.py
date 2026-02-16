import torch
import yaml
from models.frontend_conv import ConvFrontend, FrontendConfig
from models.encoder import Encoder, EncoderConfig
from models.decoder_generator import WaveformDecoder, DecoderConfig
from models.sigreg import SIGReg, SIGRegConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Load the config
    with open("configs/exp0.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize components
    mcfg = cfg["model"]
    
    frontend = ConvFrontend(FrontendConfig(**mcfg["frontend"]))
    print(f"Frontend: {count_parameters(frontend):,} params")
    
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg["encoder"]))
    print(f"Encoder: {count_parameters(encoder):,} params")
    
    latent_dim = int(mcfg["encoder"]["d_model"])

    decoder = WaveformDecoder(latent_dim, DecoderConfig(**mcfg["decoder"]))
    print(f"Decoder: {count_parameters(decoder):,} params")
    
    # Total
    total = count_parameters(frontend) + count_parameters(encoder) + count_parameters(decoder)
    print(f"-" * 30)
    print(f"TOTAL TRAINABLE PARAMETERS: {total:,}")
    print(f"-" * 30)

if __name__ == "__main__":
    main()
