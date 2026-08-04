from __future__ import annotations

import math

import pytest
import torch

from models.mhc import MHCWrapper
from models.visreg import VISReg
from schema import VISRegCfg
from training_diagnostics import mhc_diagnostics, representation_distribution_stats


def test_representation_distribution_stats_for_isotropic_axes() -> None:
    population = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    stats = representation_distribution_stats(population, prefix="latent/z")

    assert stats["latent/z/effective_rank"] == pytest.approx(2.0)
    assert stats["latent/z/isotropy_ratio"] == pytest.approx(1.0)
    assert stats["latent/z/cov_offdiag_abs_mean"] == pytest.approx(0.0)
    assert stats["latent/z/collapsed_dim_frac"] == pytest.approx(0.0)


def test_representation_distribution_stats_detects_collapsed_dimension() -> None:
    population = torch.tensor([[0.0, 1.0], [0.0, -1.0], [0.0, 0.5]])
    stats = representation_distribution_stats(population, prefix="projector/p")

    assert stats["projector/p/collapsed_dim_frac"] == pytest.approx(0.5)
    assert stats["projector/p/isotropy_ratio"] == pytest.approx(0.0)


def test_mhc_diagnostics_exposes_matrix_and_parameter_metrics() -> None:
    encoder = torch.nn.Module()
    encoder.add_module(
        "wrapper",
        MHCWrapper(
            num_streams=2,
            layer_index=0,
            sinkhorn_iters=10,
            tau=0.05,
            identity_mix=True,
        ),
    )
    stats = mhc_diagnostics(encoder)

    assert stats["mhc/enabled"] == 1.0
    assert stats["mhc/wrapper_count"] == 1.0
    assert math.isfinite(stats["mhc/wrapper/residual_alpha"])
    assert stats["mhc/wrapper/sinkhorn_max_row_sum_error"] < 1.0e-4


def test_mhc_diagnostics_marks_disabled_encoder() -> None:
    stats = mhc_diagnostics(torch.nn.Linear(2, 2))
    assert stats == {"mhc/enabled": 0.0, "mhc/wrapper_count": 0.0}


def test_visreg_exposed_terms_sum_to_unchanged_objective() -> None:
    regularizer = VISReg(VISRegCfg(num_projections=8))
    population = torch.randn(1, 16, 4, requires_grad=True)

    total, terms = regularizer.forward_with_terms(population)

    assert total.item() == pytest.approx(sum(value.item() for value in terms.values()))
    total.backward()
    assert population.grad is not None
