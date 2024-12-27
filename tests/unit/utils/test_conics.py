import torch

from ellipse_rcnn.utils.conics import (
    bbox_ellipse,
    conic_center,
    ellipse_axes,
    ellipse_angle,
    adjugate_matrix,
    unimodular_matrix,
    ellipse_to_conic_matrix,
)


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
    major_axis, minor_axis = torch.tensor([5.0, 6.0]), torch.tensor([3.0, 4.0])

    cx, cy = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    theta = torch.tensor([0.1, 0.2])

    conic_matrices = ellipse_to_conic_matrix(major_axis, minor_axis, cx, cy, theta)

    assert torch.allclose(
        torch.cat(ellipse_axes(conic_matrices)), torch.cat((major_axis, minor_axis))
    )
    assert torch.allclose(torch.cat(conic_center(conic_matrices)), torch.cat((cx, cy)))
    assert torch.allclose(ellipse_angle(conic_matrices), theta)

    # Test with circular ellipse

    major_axis, minor_axis = torch.tensor([5.0, 6.0]), torch.tensor([5.0, 6.0])

    conic_matrices = ellipse_to_conic_matrix(major_axis, minor_axis, cx, cy, theta)

    assert torch.allclose(
        torch.cat(ellipse_axes(conic_matrices)), torch.cat((major_axis, minor_axis))
    )
    assert torch.allclose(torch.cat(conic_center(conic_matrices)), torch.cat((cx, cy)))

    # Since the ellipse is circular, the angle should be 0.
    assert torch.allclose(ellipse_angle(conic_matrices), torch.tensor([0.0, 0.0]))


def test_bbox_ellipse() -> None:
    semimajor_axis, semiminor_axis = torch.tensor([5.0, 6.0]), torch.tensor([3.0, 4.0])
    cx, cy = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    theta = torch.tensor([0.0, 0.0])

    ellipse_matrices = ellipse_to_conic_matrix(
        semimajor_axis, semiminor_axis, cx, cy, theta
    )

    expected_bbox = torch.tensor([[-4, 0.0, 6.0, 6.0], [-4.0, 0.0, 8.0, 8.0]])
    calculated_bbox = bbox_ellipse(ellipse_matrices)

    assert torch.allclose(calculated_bbox, expected_bbox)

    # Example from
    # https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse

    semimajor_axis, semiminor_axis = torch.tensor([2.0]), torch.tensor([1.0])
    cx, cy = torch.tensor([0.0]), torch.tensor([0.0])
    theta = torch.tensor([0.0])

    ellipse_matrix = ellipse_to_conic_matrix(
        semimajor_axis, semiminor_axis, cx, cy, theta
    )
    expected_bbox = torch.tensor([[-2.0, -1.0, 2.0, 1.0]])

    calculated_bbox = bbox_ellipse(ellipse_matrix)

    assert torch.allclose(calculated_bbox, expected_bbox)
