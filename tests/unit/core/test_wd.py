import pytest
import torch

from ellipse_rcnn.core.wd import wasserstein_distance

from . import sample_conic_ellipses


def test_wasserstein_distance_shape_only_true() -> None:
    A1 = sample_conic_ellipses(2)
    A2 = sample_conic_ellipses(2)
    result = wasserstein_distance(A1, A2, shape_only=True)
    print("Test wasserstein_distance_shape_only_true: Result =", result)
    assert result.shape == torch.Size([2])
    assert torch.isfinite(result).all()
    assert (result >= 0).all()  # Wasserstein distance is non-negative


def test_wasserstein_distance_shape_only_false() -> None:
    A1 = sample_conic_ellipses(2)
    A2 = sample_conic_ellipses(2)
    result = wasserstein_distance(A1, A2, shape_only=False)
    print("Test wasserstein_distance_shape_only_false: Result =", result)
    assert result.shape == torch.Size([2])
    assert torch.isfinite(result).all()
    assert (result >= 0).all()


def test_wasserstein_distance_different_batch_sizes() -> None:
    A1 = sample_conic_ellipses(2)
    A2 = sample_conic_ellipses(3)
    with pytest.raises(ValueError):
        wasserstein_distance(A1, A2)


def test_wasserstein_distance_numerical_stability() -> None:
    A1 = sample_conic_ellipses(2) * 1e-8
    A2 = sample_conic_ellipses(2) * 1e-8
    result = wasserstein_distance(A1, A2)
    print("Test wasserstein_distance_numerical_stability: Result =", result)
    assert result.shape == torch.Size([2])
    assert torch.isfinite(result).all()
    assert (result >= 0).all()


def test_wasserstein_distance_identity_matrices() -> None:
    """Test when both matrices are identity matrices - should give zero distance."""
    A1 = torch.eye(3).expand(2, -1, -1)
    A2 = torch.eye(3).expand(2, -1, -1)
    result = wasserstein_distance(A1, A2)
    print("Test wasserstein_distance_identity_matrices: Result =", result)
    assert torch.allclose(result, torch.zeros(2), atol=1e-7)


def test_wasserstein_distance_highly_dissimilar_matrices() -> None:
    """Test with two very different matrices."""
    A1 = torch.tensor([[[2.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 1.0]]])
    A2 = torch.tensor([[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]]])
    result = wasserstein_distance(A1, A2)
    print("Test wasserstein_distance_highly_dissimilar_matrices: Result =", result)
    assert result.shape == torch.Size([1])
    assert torch.isfinite(result).all()
    assert result > 1.0  # Expect larger distance due to significant differences


def test_wasserstein_distance_displacement_effect() -> None:
    """Test the impact of displacement when shape_only=False."""
    A1 = torch.eye(3).expand(1, -1, -1)
    A2 = torch.eye(3).expand(1, -1, -1)
    A1[..., :2, -1] = torch.tensor([1.0, 0.0])  # Shift center
    A2[..., :2, -1] = torch.tensor([0.0, 1.0])  # Shift center
    result = wasserstein_distance(A1, A2, shape_only=False)
    shape_only_result = wasserstein_distance(A1, A2, shape_only=True)
    assert result > shape_only_result  # Displacement should increase distance
    assert torch.isclose(shape_only_result, torch.tensor(0.0), atol=1e-7)


def test_wasserstein_distance_symmetry() -> None:
    """Test that the Wasserstein distance is symmetric."""
    A1 = sample_conic_ellipses(2)
    A2 = sample_conic_ellipses(2)
    result1 = wasserstein_distance(A1, A2)
    result2 = wasserstein_distance(A2, A1)
    assert torch.allclose(result1, result2, atol=1e-7)


def test_wasserstein_distance_triangle_inequality() -> None:
    """Test that the Wasserstein distance satisfies the triangle inequality."""
    A1 = sample_conic_ellipses(1)
    A2 = sample_conic_ellipses(1)
    A3 = sample_conic_ellipses(1)

    d12 = torch.sqrt(wasserstein_distance(A1, A2))
    d23 = torch.sqrt(wasserstein_distance(A2, A3))
    d13 = torch.sqrt(wasserstein_distance(A1, A3))

    # Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
    assert d13 <= d12 + d23 + 1e-6  # Small epsilon for numerical stability


def test_wasserstein_distance_scale_effect() -> None:
    """Test the effect of scaling on the Wasserstein distance."""
    A1 = torch.eye(3).expand(1, -1, -1)
    A2 = 2 * torch.eye(3).expand(1, -1, -1)
    result = wasserstein_distance(A1, A2)
    assert result > 0
    assert torch.isfinite(result).all()


def test_wasserstein_distance_large_batch() -> None:
    """Test with a large batch of matrices to ensure scalability."""
    A1 = torch.eye(3).expand(1000, -1, -1)
    A2 = torch.eye(3).expand(1000, -1, -1)
    result = wasserstein_distance(A1, A2)
    assert result.shape == torch.Size([1000])
    assert torch.allclose(result, torch.zeros(1000), atol=1e-7)


def test_wasserstein_distance_gradient() -> None:
    """Test that gradients can be computed through the Wasserstein distance."""
    A1 = torch.randn(1, 3, 3, requires_grad=True)
    A2 = torch.randn(1, 3, 3)
    result = wasserstein_distance(A1, A2)
    result.backward()
    assert A1.grad is not None
    assert torch.isfinite(A1.grad).all()


@pytest.mark.skip
def test_wasserstein_distance_singular_matrices() -> None:
    """
    Test when one or both matrices are singular.
    Skipped as it requires special handling in conic_center.
    """
    A1 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    A2 = torch.eye(3).expand(1, -1, -1)
    result = wasserstein_distance(A1, A2)
    assert result.shape == torch.Size([1])
    assert torch.isfinite(result).all()
