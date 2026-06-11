import torch

from models.encoder import Encoder
from models.mhc import MHCWrapper, sinkhorn_log
from utils.schema import EncoderCfg, MHCCfg


def main() -> None:
    cfg = EncoderCfg(
        d_model=32,
        n_layers=4,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.1,
        cnn_module_kernel=7,
        mhc=MHCCfg(
            enabled=True,
            num_streams=2,
            start_layer=2,
            period=1,
            sinkhorn_iters=5,
            tau=0.1,
            dropout=0.0,
        ),
    )

    encoder = Encoder(in_channels=16, cfg=cfg)
    h0 = torch.randn(2, 16, 20)

    out = encoder(h0)
    assert out.shape == (2, 32, 20), out.shape

    for mod in encoder.mhc_wrappers:
        if isinstance(mod, MHCWrapper):
            h_res = sinkhorn_log(mod.H_res_logits, num_iters=mod.mhc_num_iters, tau=mod.mhc_tau)
            row_sum = h_res.sum(dim=-1)
            col_sum = h_res.sum(dim=-2)
            assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-2), row_sum
            assert torch.allclose(col_sum, torch.ones_like(col_sum), atol=1e-2), col_sum
            break

    print("smoke test passed")


if __name__ == "__main__":
    main()
