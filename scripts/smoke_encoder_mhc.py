import torch

from models.encoder import Encoder, EncoderConfig
from models.mhc import MHCConfig, MHCWrapper, sinkhorn_log


def main() -> None:
    cfg = EncoderConfig(
        d_model=32,
        n_layers=4,
        num_heads=4,
        query_head_dim=8,
        pos_head_dim=4,
        value_head_dim=8,
        feedforward_dim=64,
        dropout=0.1,
        cnn_module_kernel=7,
        pos_dim=32,
        warmup_batches=100.0,
        mhc=MHCConfig(
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
    key_padding_mask = torch.zeros(2, 20, dtype=torch.bool)
    key_padding_mask[0, -2:] = True

    out = encoder(h0, key_padding_mask=key_padding_mask)
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
