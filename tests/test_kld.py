import pytest
import torch
from ellipse_rcnn.core.kld import mv_kullback_leibler_divergence
from ellipse_rcnn.utils.conics import unimodular_matrix, ellipse_to_conic_matrix


def sample_parametric_ellipses(
    batch_size: int,
    a_range=(2.0, 5.0),
    b_range=(1.0, 3.0),
    theta_range=(0.0, 2 * torch.pi),
    xy_range=(-1.0, 1.0),
) -> tuple[torch.Tensor, ...]:
    # Semi-major and semi-minor axes
    a = torch.rand(batch_size) * (a_range[1] - a_range[0]) + a_range[0]
    b = torch.rand(batch_size) * (b_range[1] - b_range[0]) + b_range[0]
    # Ensure b <= a
    b = torch.minimum(a, b)

    # Angle in radians
    theta = torch.rand(batch_size) * (theta_range[1] - theta_range[0]) + theta_range[0]

    # Positions
    x = torch.rand(batch_size) * (xy_range[1] - xy_range[0]) + xy_range[0]
    y = torch.rand(batch_size) * (xy_range[1] - xy_range[0]) + xy_range[0]

    return a, b, x, y, theta


def sample_conic_ellipses(
    batch_size: int,
    a_range=(2.0, 5.0),
    b_range=(1.0, 3.0),
    theta_range=(0.0, 2 * torch.pi),
    xy_range=(-1.0, 1.0),
) -> torch.Tensor:
    a, b, x, y, theta = sample_parametric_ellipses(
        batch_size, a_range, b_range, theta_range, xy_range
    )
    return ellipse_to_conic_matrix(a, b, x, y, theta)


def test_mv_kl_divergence_shape_only_true() -> None:
    A1 = sample_conic_ellipses(2)
    A2 = sample_conic_ellipses(2)
    result = mv_kullback_leibler_divergence(A1, A2, shape_only=True)
    print("Test mv_kl_divergence_shape_only_true: Result =", result)
    assert result.shape == torch.Size([2])
    assert torch.isfinite(result).all()


def test_mv_kl_divergence_shape_only_false() -> None:
    A1 = sample_conic_ellipses(2)
    A2 = sample_conic_ellipses(2)

    result = mv_kullback_leibler_divergence(A1, A2, shape_only=False)
    print("Test mv_kl_divergence_shape_only_false: Result =", result)
    assert result.shape == torch.Size([2])
    assert torch.isfinite(result).all()


def test_mv_kl_divergence_different_batch_sizes() -> None:
    A1 = sample_conic_ellipses(2)
    A2 = sample_conic_ellipses(3)
    with pytest.raises(ValueError):
        mv_kullback_leibler_divergence(A1, A2)


def test_mv_kl_divergence_numerical_stability() -> None:
    A1 = sample_conic_ellipses(2) * 1e-8
    A2 = sample_conic_ellipses(2) * 1e-8
    result = mv_kullback_leibler_divergence(A1, A2)
    print("Test mv_kl_divergence_numerical_stability: Result =", result)
    assert result.shape == torch.Size([2])
    assert torch.isfinite(result).all()


def test_mv_kl_divergence_unimodular_matrices() -> None:
    A1 = unimodular_matrix(torch.eye(3))
    A2 = unimodular_matrix(torch.eye(3))
    A1 = A1.expand(2, -1, -1)
    A2 = A2.expand(2, -1, -1)
    result = mv_kullback_leibler_divergence(A1, A2)
    print("Test mv_kl_divergence_unimodular_matrices: Result =", result)
    assert torch.allclose(result, torch.tensor([0.0, 0.0]), atol=1e-7)


def test_mv_kl_divergence_identity_matrices() -> None:
    """
    Test when both matrices are identity matrices.
    """
    A1 = torch.eye(3).expand(2, -1, -1)  # Batch of two identity matrices
    A2 = torch.eye(3).expand(2, -1, -1)
    result = mv_kullback_leibler_divergence(A1, A2)
    print("Test mv_kl_divergence_identity_matrices: Result =", result)
    assert torch.allclose(result, torch.zeros(2), atol=1e-7)


def test_mv_kl_divergence_highly_dissimilar_matrices() -> None:
    """
    Test with two very different matrices.
    """
    A1 = torch.tensor([[[2.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 1.0]]])
    A2 = torch.tensor([[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]]])

    result = mv_kullback_leibler_divergence(A1, A2)
    print("Test mv_kl_divergence_highly_dissimilar_matrices: Result =", result)
    assert result.shape == torch.Size([1])
    assert torch.isfinite(result).all()
    assert result > 1.0  # Expect a larger divergence due to large differences


def test_mv_kl_divergence_displacement_effect() -> None:
    """
    Test the impact of displacement (mismatch in centers) when shape_only=False.
    """
    A1 = torch.eye(3).expand(1, -1, -1)
    A2 = torch.eye(3).expand(1, -1, -1)
    # Introduce displacement in the centers (A1 and A2 have the same shape)
    A1[..., :2, -1] = torch.tensor([1.0, 0.0])  # Shift center
    A2[..., :2, -1] = torch.tensor([0.0, 1.0])  # Shift center
    result = mv_kullback_leibler_divergence(A1, A2, shape_only=False)
    assert result.shape == torch.Size([1])
    assert torch.isfinite(result).all()
    assert result > 0.0  # KL divergence should account for the center displacement


def test_mv_kl_divergence_small_vs_large_matrices() -> None:
    """
    Test with one small and one large ellipse matrix (checks numerical stability).
    """
    A1 = torch.eye(3).expand(1, -1, -1) * 1e-3  # Small matrix
    A2 = torch.eye(3).expand(1, -1, -1) * 1e3  # Large matrix
    result = mv_kullback_leibler_divergence(A1, A2)
    assert result.shape == torch.Size([1])
    assert torch.isfinite(result).all()
    assert result > 0.0  # Expect a larger divergence due to scale differences


def test_mv_kl_divergence_singular_matrices() -> None:
    """
    Test when one or both matrices are singular (determinants are very close to zero).
    """
    A1 = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )  # Singular matrix
    A2 = torch.eye(3).expand(1, -1, -1)
    result = mv_kullback_leibler_divergence(A1, A2)
    assert result.shape == torch.Size([1])
    assert torch.isfinite(result).all()


def test_mv_kl_divergence_diagonal_matrices() -> None:
    """
    Test when matrices are symmetric diagonals, which simplify computation.
    """
    A1 = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]])
    A2 = torch.tensor([[[4.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]]])
    result = mv_kullback_leibler_divergence(A1, A2)
    assert result.shape == torch.Size([1])
    assert torch.isfinite(result).all()
    assert result > 0.0  # KL divergence accounts for diagonal differences


def test_mv_kl_divergence_large_batch() -> None:
    """
    Test with a large batch of matrices to ensure scalability.
    """
    A1 = torch.eye(3).expand(1000, -1, -1)  # Batch of 1000 identity matrices
    A2 = torch.eye(3).expand(1000, -1, -1)
    result = mv_kullback_leibler_divergence(A1, A2)
    assert result.shape == torch.Size([1000])
    assert torch.allclose(result, torch.zeros(1000), atol=1e-7)
