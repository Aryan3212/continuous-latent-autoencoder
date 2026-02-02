import torch

from models.sigreg import SIGReg, SIGRegConfig


def test_sigreg_gaussian_low_vs_collapsed_high():
    torch.manual_seed(0)
    cfg = SIGRegConfig(num_slices=64, t_max=3.0, n_points=9, reduction="mean")
    sigreg = SIGReg(dim=8, cfg=cfg)

    z_gauss = torch.randn(4, 8, 16)
    loss_gauss, _ = sigreg(z_gauss, step=0)

    z_collapsed = torch.zeros(4, 8, 16)
    loss_collapsed, _ = sigreg(z_collapsed, step=0)

    assert loss_collapsed > loss_gauss


if __name__ == "__main__":
    test_sigreg_gaussian_low_vs_collapsed_high()
    print("sigreg test passed")
