import torch

from ellipse_rcnn.core.ga import gaussian_angle_distance
from . import sample_conic_ellipses


def test_gaussian_angle_distance_random_conic_ellipses():
    """Test with random conic ellipses sampled using `sample_conic_ellipses`."""
    batch_size = 10
    A1 = sample_conic_ellipses(batch_size)
    A2 = sample_conic_ellipses(batch_size)

    result = gaussian_angle_distance(A1, A2)
    assert result.shape == (batch_size,), "Output shape does not match the batch size."
    assert ~torch.isnan(result).any(), "NaN values detected in the output."
    assert torch.all(result >= 0), "Distances should be non-negative."


def test_gaussian_angle_distance_same_conic_ellipses():
    """Test with identical conic ellipses to ensure distance is zero."""
    batch_size = 10
    A1 = sample_conic_ellipses(batch_size)

    # Use the same ellipses for A2
    A2 = A1.clone()

    result = gaussian_angle_distance(A1, A2)
    assert ~torch.isnan(result).any(), "NaN values detected in the output."
    assert torch.allclose(
        result, torch.zeros_like(result), atol=1e-5
    ), "Distance between identical conic ellipses should be zero."


def test_gaussian_angle_distance_varied_batch_sizes():
    """Test with varying batch sizes."""
    batch_sizes = [10, 100, 1000]
    for batch_size in batch_sizes:
        A1 = sample_conic_ellipses(batch_size)
        A2 = sample_conic_ellipses(batch_size)

        result = gaussian_angle_distance(A1, A2)
        assert result.shape == (
            batch_size,
        ), f"Output shape does not match the batch size: {batch_size}."
        assert ~torch.isnan(result).any(), "NaN values detected in the output."
        assert torch.all(result >= 0), "Distances should be non-negative."


def test_gaussian_angle_distance_numerical_stability():
    """Test numerical stability with extreme conic parameters."""
    batch_size = 10

    # Generate conic ellipses with very small parameters
    A1_small = sample_conic_ellipses(
        batch_size, a_range=(1e-6, 1e-5), b_range=(1e-6, 1e-5)
    )
    A2_small = sample_conic_ellipses(
        batch_size, a_range=(1e-6, 1e-5), b_range=(1e-6, 1e-5)
    )

    # Generate conic ellipses with very large parameters
    A1_large = sample_conic_ellipses(batch_size, a_range=(1e3, 1e4), b_range=(1e3, 1e4))
    A2_large = sample_conic_ellipses(batch_size, a_range=(1e3, 1e4), b_range=(1e3, 1e4))

    result_small = gaussian_angle_distance(A1_small, A2_small)
    result_large = gaussian_angle_distance(A1_large, A2_large)

    assert torch.all(
        result_small >= 0
    ), "Distances with small parameters should be non-negative."
    assert ~torch.isnan(result_small).any(), "NaN values detected in the output."
    assert ~torch.isnan(result_large).any(), "NaN values detected in the output."
    assert torch.all(
        result_large >= 0
    ), "Distances with large parameters should be non-negative."


def test_gaussian_angle_distance_batch_processing():
    """Test batch behavior and accuracy for larger sample sizes."""
    batch_size = 1000
    A1 = sample_conic_ellipses(batch_size)
    A2 = sample_conic_ellipses(batch_size)

    result = gaussian_angle_distance(A1, A2)
    assert result.shape == (
        batch_size,
    ), "Batch output shape does not match the input size."
    assert ~torch.isnan(result).any(), "NaN values detected in the output."
    assert torch.all(result >= 0), "Distances should be non-negative."
