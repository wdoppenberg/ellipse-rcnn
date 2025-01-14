import torch

from ellipse_rcnn.core.ops import (
    bbox_ellipse_matrix,
    ellipse_center,
    ellipse_axes,
    ellipse_angle,
    ellipse_to_conic_matrix,
    ellipse_area,
    bbox_ellipse,
)
from ellipse_rcnn.utils.mat import adjugate_matrix, unimodular_matrix
from tests.unit.core import sample_conic_ellipses


def test_adjugate_matrix() -> None:
    """
    Test adjugate matrix. Used https://www.geeksforgeeks.org/how-to-find-cofactor-of-a-matrix-using-numpy
    as a reference.
    """
    input_matrix = torch.tensor([[1, 9, 3], [2, 5, 4], [3, 7, 8]], dtype=torch.float32)

    expected_output = torch.tensor(
        [[12.0, -51.0, 21.0], [-4.0, -1.0, 2.0], [-1.0, 20.0, -13.0]],
        dtype=torch.float32,
    )

    output = adjugate_matrix(input_matrix)

    assert torch.allclose(output, expected_output)

    input_matrices = torch.cat(
        (input_matrix[None, ...], input_matrix[None, ...]), dim=0
    )

    expected_outputs = torch.cat(
        (expected_output[None, ...], expected_output[None, ...]), dim=0
    )

    outputs = adjugate_matrix(input_matrices)

    assert torch.allclose(outputs, expected_outputs)


def test_unimodular_matrix() -> None:
    """
    Test adjugate matrix
    """
    input_matrix = torch.tensor(
        [[-3, 2, -5], [-5, 12, -1], [4, -6, 2]], dtype=torch.float32
    )

    assert torch.isclose(torch.det(unimodular_matrix(input_matrix)), torch.tensor(1.0))

    input_matrices = torch.cat(
        (input_matrix[None, ...], input_matrix[None, ...]), dim=0
    )

    assert torch.allclose(
        torch.det(unimodular_matrix(input_matrices)), torch.tensor(1.0)
    )


def test_ellipse_conic_conversion() -> None:
    """
    Test ellipse to conic matrix by converting conic to ellipse and back.
    """
    a, b = torch.tensor([5.0, 6.0]), torch.tensor([3.0, 4.0])

    cx, cy = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    theta = torch.tensor([0.1, 0.2])

    conic_matrices = ellipse_to_conic_matrix(a=a, b=b, x=cx, y=cy, theta=theta)

    assert torch.allclose(torch.cat(ellipse_axes(conic_matrices)), torch.cat((a, b)))
    assert torch.allclose(
        torch.cat(ellipse_center(conic_matrices)), torch.cat((cx, cy))
    )
    assert torch.allclose(ellipse_angle(conic_matrices), theta)

    # Test with circular ellipse

    a, b = torch.tensor([5.0, 6.0]), torch.tensor([5.0, 6.0])

    conic_matrices = ellipse_to_conic_matrix(a=a, b=b, x=cx, y=cy, theta=theta)

    assert torch.allclose(torch.cat(ellipse_axes(conic_matrices)), torch.cat((a, b)))
    assert torch.allclose(
        torch.cat(ellipse_center(conic_matrices)), torch.cat((cx, cy))
    )

    # Since the ellipse is circular, the angle should be 0.
    assert torch.allclose(ellipse_angle(conic_matrices), torch.tensor([0.0, 0.0]))


def test_bbox_ellipse() -> None:
    a, b = torch.tensor([5.0, 6.0]), torch.tensor([3.0, 4.0])
    cx, cy = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    theta = torch.tensor([0.0, 0.0])
    ellipse = torch.stack([a, b, cx, cy, theta], dim=1)

    expected_bbox = torch.tensor([[-4.0, 0.0, 6.0, 6.0], [-4.0, 0.0, 8.0, 8.0]])
    calculated_bbox = bbox_ellipse(ellipse)

    assert torch.allclose(calculated_bbox, expected_bbox)

    # Example from
    # https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse

    a, b = torch.tensor([2.0]), torch.tensor([1.0])
    cx, cy = torch.tensor([0.0]), torch.tensor([0.0])
    theta = torch.tensor([0.0])

    ellipse_matrix = ellipse_to_conic_matrix(a=a, b=b, x=cx, y=cy, theta=theta)
    expected_bbox = torch.tensor([[-2.0, -1.0, 2.0, 1.0]])

    calculated_bbox = bbox_ellipse_matrix(ellipse_matrix)

    assert torch.allclose(calculated_bbox, expected_bbox)


def test_simple_ellipse() -> None:
    """
    Simple test for a basic ellipse without rotation.
    """
    major_axis = torch.tensor([4.0])
    minor_axis = torch.tensor([2.0])
    cx = torch.tensor([0.0])
    cy = torch.tensor([0.0])
    theta = torch.tensor([0.0])  # No rotation

    conic_matrix = ellipse_to_conic_matrix(
        a=major_axis, b=minor_axis, x=cx, y=cy, theta=theta
    )

    # Check axes retrieval
    a_out, b_out = ellipse_axes(conic_matrix)
    assert torch.allclose(a_out, major_axis)
    assert torch.allclose(b_out, minor_axis)

    # Check center retrieval
    cx_out, cy_out = ellipse_center(conic_matrix)
    assert torch.allclose(cx_out, cx)
    assert torch.allclose(cy_out, cy)

    # Check angle retrieval
    theta_out = ellipse_angle(conic_matrix)
    assert torch.allclose(theta_out, theta)


def test_unit_circle() -> None:
    """
    Very simple test for a unit circle centered at origin.
    """
    major_axis = torch.tensor([1.0])
    minor_axis = torch.tensor([1.0])
    cx = torch.tensor([0.0])
    cy = torch.tensor([0.0])
    theta = torch.tensor([0.0])  # No rotation

    conic_matrix = ellipse_to_conic_matrix(
        a=major_axis, b=minor_axis, x=cx, y=cy, theta=theta
    )

    # Check outputs
    a_out, b_out = ellipse_axes(conic_matrix)
    cx_out, cy_out = ellipse_center(conic_matrix)

    assert torch.allclose(a_out, major_axis)
    assert torch.allclose(b_out, minor_axis)
    assert torch.allclose(cx_out, cx)
    assert torch.allclose(cy_out, cy)

    # Angle should be zero for a circle
    theta_out = ellipse_angle(conic_matrix)
    assert torch.allclose(theta_out, theta)


def test_shifted_ellipse() -> None:
    """
    Simple test for shifted ellipse without rotation.
    """
    major_axis = torch.tensor([5.0])
    minor_axis = torch.tensor([3.0])
    cx = torch.tensor([2.0])
    cy = torch.tensor([-1.0])
    theta = torch.tensor([0.0])  # No rotation

    conic_matrix = ellipse_to_conic_matrix(
        a=major_axis, b=minor_axis, x=cx, y=cy, theta=theta
    )

    # Check outputs
    a_out, b_out = ellipse_axes(conic_matrix)
    cx_out, cy_out = ellipse_center(conic_matrix)

    assert torch.allclose(a_out, major_axis)
    assert torch.allclose(b_out, minor_axis)
    assert torch.allclose(cx_out, cx)
    assert torch.allclose(cy_out, cy)

    # Rotation should still be zero
    theta_out = ellipse_angle(conic_matrix)
    assert torch.allclose(theta_out, theta)


def test_rotated_circle() -> None:
    """
    Simple test for a rotated circle (should still behave like a circle).
    """
    major_axis = torch.tensor([2.0])
    minor_axis = torch.tensor([2.0])  # Same as major axis
    cx = torch.tensor([0.0])
    cy = torch.tensor([0.0])
    theta = torch.tensor([torch.pi / 4])  # 45 degree rotation

    conic_matrix = ellipse_to_conic_matrix(
        a=major_axis, b=minor_axis, x=cx, y=cy, theta=theta
    )

    # Check outputs
    a_out, b_out = ellipse_axes(conic_matrix)
    cx_out, cy_out = ellipse_center(conic_matrix)

    assert torch.allclose(a_out, major_axis)
    assert torch.allclose(b_out, minor_axis)
    assert torch.allclose(cx_out, cx)
    assert torch.allclose(cy_out, cy)

    # Since it's a circle, the angle should resolve to 0
    theta_out = ellipse_angle(conic_matrix)
    assert torch.allclose(theta_out % torch.pi, torch.tensor(0.0))


def test_degenerate_ellipse() -> None:
    """
    Test handling of a degenerate ellipse where axes are zero.
    """
    major_axis = torch.tensor([0.0])
    minor_axis = torch.tensor([0.0])
    cx, cy = torch.tensor([0.0]), torch.tensor([0.0])
    theta = torch.tensor([0.5])  # some angle

    conic_matrix = ellipse_to_conic_matrix(
        a=major_axis, b=minor_axis, x=cx, y=cy, theta=theta
    )

    # Conic matrix should be a 3x3 zero matrix
    expected = torch.zeros(3, 3, dtype=torch.float32)
    assert torch.allclose(conic_matrix, expected)


def test_circular_ellipse() -> None:
    """
    Test circular ellipse conversion with non-zero center.
    """
    major_axis = torch.tensor([5.0, 10.0])
    minor_axis = torch.tensor([5.0, 10.0])  # Same as major axis
    cx, cy = torch.tensor([3.0, -5.0]), torch.tensor([4.0, 7.0])
    theta = torch.tensor([0.0, torch.pi / 4])  # 0 rotation and 45 degrees

    conic_matrices = ellipse_to_conic_matrix(
        a=major_axis, b=minor_axis, x=cx, y=cy, theta=theta
    )

    # Since it's a circular ellipse, axis lengths and center should be preserved
    assert torch.allclose(
        torch.cat(ellipse_center(conic_matrices)), torch.cat((cx, cy))
    )
    assert torch.allclose(
        torch.cat(ellipse_axes(conic_matrices)), torch.cat((major_axis, minor_axis))
    )
    assert torch.allclose(ellipse_angle(conic_matrices), torch.tensor([0.0, 0.0]))


def test_rotated_ellipse() -> None:
    """
    Test ellipses with varying eccentricities and rotations.
    """
    major_axis = torch.tensor([6.0, 10.0, 8.0])
    minor_axis = torch.tensor([4.0, 5.0, 3.0])  # Different eccentricities
    cx, cy = torch.tensor([0.0, -10.0, 1.0]), torch.tensor([2.0, 4.0, -2.0])
    theta = torch.tensor(
        [0.0, torch.pi / 6, torch.pi / 3]
    )  # No rotation, 30, 60 degrees

    conic_matrices = ellipse_to_conic_matrix(
        a=major_axis, b=minor_axis, x=cx, y=cy, theta=theta
    )

    # Test the recovering of ellipse parameters
    assert torch.allclose(
        torch.cat(ellipse_axes(conic_matrices)), torch.cat((major_axis, minor_axis))
    )
    assert torch.allclose(
        torch.cat(ellipse_center(conic_matrices)), torch.cat((cx, cy))
    )
    assert torch.allclose(ellipse_angle(conic_matrices), theta)


def test_large_ellipse() -> None:
    """
    Test ellipse conversion with very large axes.
    """
    major_axis = torch.tensor([1000.0])
    minor_axis = torch.tensor([500.0])
    cx, cy = torch.tensor([0.0]), torch.tensor([0.0])
    theta = torch.tensor([torch.pi / 4])  # 45 degrees rotation

    conic_matrix = ellipse_to_conic_matrix(
        a=major_axis, b=minor_axis, x=cx, y=cy, theta=theta
    )

    # Test the recovering of ellipse parameters
    a_out, b_out = ellipse_axes(conic_matrix)
    cx_out, cy_out = ellipse_center(conic_matrix)
    theta_out = ellipse_angle(conic_matrix)

    assert torch.allclose(a_out, major_axis)
    assert torch.allclose(b_out, minor_axis)
    assert torch.allclose(cx_out, cx)
    assert torch.allclose(cy_out, cy)
    assert torch.allclose(theta_out, theta)


def test_ellipse_extreme_rotation() -> None:
    """
    Test ellipses converted at extreme rotations.
    """
    major_axis = torch.tensor([7.0])
    minor_axis = torch.tensor([3.0])
    cx, cy = torch.tensor([2.0]), torch.tensor([-3.0])
    theta = torch.tensor([torch.pi])  # 180 degree rotation

    conic_matrix = ellipse_to_conic_matrix(
        a=major_axis, b=minor_axis, x=cx, y=cy, theta=theta
    )

    # Test recovering of ellipse parameters
    a_out, b_out = ellipse_axes(conic_matrix)
    cx_out, cy_out = ellipse_center(conic_matrix)
    theta_out = ellipse_angle(conic_matrix)

    assert torch.allclose(a_out, major_axis)
    assert torch.allclose(b_out, minor_axis)
    assert torch.allclose(cx_out, cx)
    assert torch.allclose(cy_out, cy)

    # Since 180 degrees is equivalent to 0 in modulus, angle should be approximately 0
    assert torch.allclose(theta_out % torch.pi, torch.tensor(0.0), atol=1e-7)


def test_ellipse_area():
    # Test case: Single ellipse
    ellipses = torch.tensor([[3.0, 2.0, 0.0, 0.0, 0.0]])  # [a, b, cx, cy, theta]
    expected_area = torch.tensor([3.0 * 2.0 * torch.pi])
    computed_area = ellipse_area(ellipses)
    assert torch.allclose(
        computed_area, expected_area
    ), f"Expected {expected_area}, got {computed_area}"

    # Test case: Multiple ellipses
    ellipses = torch.tensor(
        [
            [3.0, 2.0, 0.0, 0.0, 0.0],
            [5.0, 1.0, 1.0, 2.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    expected_areas = torch.tensor(
        [
            3.0 * 2.0 * torch.pi,
            5.0 * 1.0 * torch.pi,
            1.0 * 1.0 * torch.pi,
        ]
    )
    computed_areas = ellipse_area(ellipses)
    assert torch.allclose(
        computed_areas, expected_areas
    ), f"Expected {expected_areas}, got {computed_areas}"

    # Test case: Degenerate ellipse
    ellipses = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    expected_area = torch.tensor([0.0])
    computed_area = ellipse_area(ellipses)
    assert torch.allclose(
        computed_area, expected_area
    ), f"Expected {expected_area}, got {computed_area}"


def test_single_ellipse_center():
    ellipses_mat = sample_conic_ellipses(1)
    cx, cy = ellipse_center(ellipses_mat)

    assert cx.shape == (1,)
    assert cy.shape == (1,)
