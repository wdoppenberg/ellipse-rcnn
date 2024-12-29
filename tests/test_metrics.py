import pytest
import torch
from ellipse_rcnn.core.kld import mv_kullback_leibler_divergence
from ellipse_rcnn.utils.conics import unimodular_matrix


def test_mv_kl_divergence_shape_only_true() -> None:
    A1 = torch.rand(2, 3, 3)
    A2 = torch.rand(2, 3, 3)
    result = mv_kullback_leibler_divergence(A1, A2, shape_only=True)
    assert result.shape == torch.Size([2])
    assert torch.isfinite(result).all()


def test_mv_kl_divergence_shape_only_false() -> None:
    A1 = torch.rand(1, 3, 3)
    A2 = torch.rand(1, 3, 3)
    result = mv_kullback_leibler_divergence(A1, A2, shape_only=False)
    assert result.shape == torch.Size([1])
    assert torch.isfinite(result).all()


def test_mv_kl_divergence_different_batch_sizes() -> None:
    A1 = torch.rand(3, 3, 3)
    A2 = torch.rand(1, 3, 3)
    with pytest.raises(RuntimeError):
        mv_kullback_leibler_divergence(A1, A2)


def test_mv_kl_divergence_numerical_stability() -> None:
    A1 = torch.eye(3).expand(2, -1, -1) * 1e-8
    A2 = torch.eye(3).expand(2, -1, -1) * 1e-8
    result = mv_kullback_leibler_divergence(A1, A2)
    assert result.shape == torch.Size([2])
    assert torch.isfinite(result).all()


def test_mv_kl_divergence_unimodular_matrices() -> None:
    A1 = unimodular_matrix(torch.eye(2))
    A2 = unimodular_matrix(torch.eye(2))
    A1 = A1.expand(2, -1, -1)
    A2 = A2.expand(2, -1, -1)
    result = mv_kullback_leibler_divergence(A1, A2)
    assert torch.allclose(result, torch.tensor([0.0, 0.0]), atol=1e-7)
