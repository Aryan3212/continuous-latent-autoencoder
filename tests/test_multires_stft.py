import torch

from losses.multires_stft import MultiResSTFTConfig, MultiResSTFTLoss


def test_multires_stft_zero_on_identical():
    cfg = MultiResSTFTConfig(fft_sizes=[64, 128], hop_ratio=0.25, win_ratio=1.0)
    loss_fn = MultiResSTFTLoss(cfg)
    x = torch.randn(2, 1, 256)
    loss, _ = loss_fn(x, x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_multires_stft_nonzero_on_perturb():
    cfg = MultiResSTFTConfig(fft_sizes=[64], hop_ratio=0.25, win_ratio=1.0)
    loss_fn = MultiResSTFTLoss(cfg)
    x = torch.randn(1, 1, 256)
    x_hat = x + 0.01 * torch.randn_like(x)
    loss, _ = loss_fn(x_hat, x)
    assert loss > 0.0


if __name__ == "__main__":
    test_multires_stft_zero_on_identical()
    test_multires_stft_nonzero_on_perturb()
    print("multires stft tests passed")
