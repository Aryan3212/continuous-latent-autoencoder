import torch

from models.sigreg import SIGReg
from utils.schema import SIGRegCfg


def test_sigreg_gaussian_low_vs_collapsed_high():
    torch.manual_seed(0)
    cfg = SIGRegCfg(num_slices=64, t_max=5.0, n_points=9)
    sigreg = SIGReg(dim=8, cfg=cfg)

    z_gauss = torch.randn(64, 8)
    loss_gauss = sigreg(z_gauss, step=0)

    z_collapsed = torch.zeros(64, 8)
    loss_collapsed = sigreg(z_collapsed, step=0)

    assert loss_collapsed > loss_gauss


if __name__ == "__main__":
    test_sigreg_gaussian_low_vs_collapsed_high()
    print("sigreg test passed")
