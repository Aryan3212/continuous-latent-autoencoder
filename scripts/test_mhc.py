import torch
import torch.nn as nn
from models.mhc import MHCWrapper

class MockBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Linear(dim, dim)
    
    def forward(self, x, pos_emb, chunk_size, attn_mask, src_key_padding_mask):
        return self.layer(x)

def test_mhc_smoke():
    dim = 64
    num_streams = 4
    batch_size = 2
    seq_len = 10
    
    branch = MockBranch(dim)
    mhc = MHCWrapper(
        branch=branch, 
        dim=dim, 
        num_streams=num_streams, 
        layer_index=0, 
        sinkhorn_iters=10, 
        tau=0.05,
        identity_mix=True,
        alpha_init=0.01
    )
    
    residuals = torch.randn(num_streams, seq_len, batch_size, dim)
    
    out = mhc(
        residuals,
        pos_emb=None,
        chunk_size=-1,
        attn_mask=None,
        src_key_padding_mask=None
    )
    
    print("Output shape:", out.shape)
    assert out.shape == residuals.shape
    print("MHC smoke test passed!")

from models.encoder import Encoder, EncoderConfig

def test_encoder_integration():
    print("Testing Encoder integration...")
    cfg = EncoderConfig(
        d_model=64,
        n_layers=2,
        mhc={
            "enabled": True,
            "num_streams": 2,
            "identity_mix": True,
            "alpha_init": 0.01
        }
    )
    encoder = Encoder(in_channels=64, cfg=cfg)
    x = torch.randn(2, 64, 50) # B, C, T
    out = encoder(x)
    print("Encoder output shape:", out.shape)
    assert out.shape == (2, 64, 50) # Transposed internally
    print("Encoder integration test passed!")

if __name__ == "__main__":
    test_mhc_smoke()
    test_encoder_integration()
