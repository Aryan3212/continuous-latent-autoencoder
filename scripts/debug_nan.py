import torch
import torch.nn as nn
from utils.config import load_config
from models.frontend_conv import ConvFrontend, FrontendConfig
from models.encoder import Encoder, EncoderConfig, Bottleneck
from models.decoder_generator import WaveformDecoder, DecoderConfig
from models.sigreg import SIGReg, SIGRegConfig
from losses.multires_stft import MultiResSTFTLoss, MultiResSTFTConfig

def check(name, tensor):
    if torch.isnan(tensor).any():
        print(f"!!! {name} contains NaN")
        return True
    return False

def _lejepa_loss(center: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    return (center - view).pow(2).mean()

def main():
    cfg = load_config("configs/train_v1.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    
    frontend = ConvFrontend(FrontendConfig(**mcfg["frontend"])).to(device)
    encoder = Encoder(frontend.out_channels, EncoderConfig(**mcfg["encoder"])).to(device)
    bottleneck = Bottleneck(mcfg["encoder"]["d_model"], mcfg["bottleneck"]["latent_dim"]).to(device)
    decoder = WaveformDecoder(mcfg["bottleneck"]["latent_dim"], DecoderConfig(**mcfg["decoder"])).to(device)
    sigreg_cfg = cfg["loss"]["sigreg"].copy()
    if "weight" in sigreg_cfg: del sigreg_cfg["weight"]
    sigreg = SIGReg(mcfg["bottleneck"]["latent_dim"], SIGRegConfig(**sigreg_cfg)).to(device)
    stft = MultiResSTFTLoss(MultiResSTFTConfig(**cfg["loss"]["stft"])).to(device)

    model = nn.ModuleDict({"frontend": frontend, "encoder": encoder, "bottleneck": bottleneck, "decoder": decoder, "sigreg": sigreg})
    
    # Fake batch
    wav = torch.randn(2, 1, 32000).to(device)
    
    print("Running Forward...")
    h0 = frontend(wav)
    if check("frontend output", h0): return
    
    hE = encoder(h0)
    if check("encoder output", hE): return
    
    z = bottleneck(hE)
    if check("bottleneck output", z): return
    
    x_hat = decoder(z, target_len=32000)
    if check("decoder output", x_hat): return
    
    print("Running Losses...")
    l_stft, _ = stft(x_hat, wav)
    if check("stft loss", l_stft): return
    
    l_jepa = _lejepa_loss(z, z)
    if check("jepa loss", l_jepa): return
    
    l_sig, _ = sigreg(z)
    if check("sigreg loss", l_sig): return
    
    print("No NaNs found in forward pass! Checking backward...")
    loss = l_stft + l_jepa + l_sig
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"!!! Gradient for {name} is NaN")
            return

    print("Success! No NaNs found in forward or backward pass.")

if __name__ == "__main__":
    main()
