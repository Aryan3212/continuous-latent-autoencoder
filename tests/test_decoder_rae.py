import torch

from models.decoder_generator import DecoderConfig, WaveformDecoder


def test_decoder_length_and_grad():
    cfg = DecoderConfig(
        channels=32,
        up_strides=[2, 2],
        up_kernels=[4, 4],
        res_blocks_per_up=1,
        res_dilations=[1, 3],
        film_hidden=16,
        latent_norm=True,
        latent_norm_eps=1.0e-5,
    )
    dec = WaveformDecoder(latent_dim=8, cfg=cfg)
    mean = torch.zeros(8)
    var = torch.ones(8)
    dec.set_latent_stats(mean, var)

    z = torch.randn(2, 8, 5, requires_grad=True)
    x = dec(z, target_len=40)
    assert x.shape == (2, 1, 40)
    loss = x.abs().mean()
    loss.backward()
    assert z.grad is not None


if __name__ == "__main__":
    test_decoder_length_and_grad()
    print("decoder test passed")
