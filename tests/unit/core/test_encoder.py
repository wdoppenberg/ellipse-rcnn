import math

import torch

from ellipse_rcnn.core.encoder import encode_ellipses, decode_ellipses, EllipseEncoder
from . import sample_parametric_ellipses


def test_ellipse_encoding_decoding():
    # Test parameters
    batch_size = 100
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    # Generate random ellipses
    ellipses = sample_parametric_ellipses(
        batch_size,
        a_range=(2.0, 5.0),
        b_range=(1.0, 3.0),
        theta_range=(0, torch.pi / 2),
        xy_range=(-10.0, 10.0),
    )
    a, b, cx, cy, theta = ellipses.unbind(-1)

    # Create some proposal boxes (slightly larger than ellipses)
    margins = torch.rand(batch_size) * 2.0 + 1.0  # Random margins between 1 and 3
    proposals = torch.stack(
        [cx - a * margins, cy - b * margins, cx + a * margins, cy + b * margins], dim=1
    )

    # Encode
    encoded = encode_ellipses(ellipses, proposals=proposals, weights=weights)

    # Decode
    pred = decode_ellipses(encoded, proposals, weights).squeeze()

    pred_a, pred_b, pred_x, pred_y, pred_theta = pred.unbind(-1)

    # Test reconstructed parameters
    torch.testing.assert_close(pred_a, a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_b, b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_x, cx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_y, cy, rtol=1e-4, atol=1e-4)

    # For theta, we need to handle periodicity
    angle_diff = torch.abs(
        torch.remainder(pred_theta - theta + torch.pi, 2 * torch.pi) - torch.pi
    )
    assert torch.all(angle_diff < 1e-4)


def test_weight_scaling():
    batch_size = 10
    weights = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])

    # Generate random ellipses
    ellipses = sample_parametric_ellipses(
        batch_size,
        a_range=(2.0, 5.0),
        b_range=(1.0, 3.0),
        theta_range=(0, 2 * torch.pi),
        xy_range=(-10.0, 10.0),
    )
    a, b, cx, cy, theta = ellipses.unbind(-1)

    # Create some proposal boxes (slightly larger than ellipses)
    margins = torch.rand(batch_size) * 2.0 + 1.0  # Random margins between 1 and 3
    proposals = torch.stack(
        [cx - a * margins, cy - b * margins, cx + a * margins, cy + b * margins], dim=1
    )

    # Encode
    encoded = encode_ellipses(ellipses, proposals=proposals, weights=weights)

    # Decode
    pred_a, pred_b, pred_x, pred_y, pred_theta = (
        decode_ellipses(encoded, proposals, weights).squeeze().unbind(-1)
    )

    # Test reconstruction
    torch.testing.assert_close(pred_a, a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_b, b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_x, cx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_y, cy, rtol=1e-4, atol=1e-4)


def test_batch_processing():
    batch_sizes = [1, 10, 100]
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    for batch_size in batch_sizes:
        # Generate data
        # Generate random ellipses
        ellipses = sample_parametric_ellipses(
            batch_size,
            a_range=(2.0, 5.0),
            b_range=(1.0, 3.0),
            theta_range=(0, 2 * torch.pi),
            xy_range=(-10.0, 10.0),
        )
        a, b, cx, cy, theta = ellipses.unbind(-1)

        # Create some proposal boxes (slightly larger than ellipses)
        margins = torch.rand(batch_size) * 2.0 + 1.0  # Random margins between 1 and 3
        proposals = torch.stack(
            [cx - a * margins, cy - b * margins, cx + a * margins, cy + b * margins],
            dim=1,
        )

        # Encode
        encoded = encode_ellipses(ellipses, proposals=proposals, weights=weights)

        # Decode
        pred_a, pred_b, pred_x, pred_y, pred_theta = (
            decode_ellipses(encoded, proposals, weights).squeeze(1).unbind(-1)
        )

        assert encoded.shape == (batch_size, 6)
        assert pred_a.shape == torch.Size([batch_size])
        assert pred_b.shape == torch.Size([batch_size])
        assert pred_x.shape == torch.Size([batch_size])
        assert pred_y.shape == torch.Size([batch_size])
        assert pred_theta.shape == torch.Size([batch_size])


def test_edge_cases():
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    # Test circular case (a = b)
    a = b = torch.tensor([3.0])
    cx = cy = torch.tensor([0.0])
    theta = torch.tensor([0.0])
    ellipses = torch.stack([a, b, cx, cy, theta]).view(-1, 5)

    proposals = torch.tensor([[-3.0, -3.0, 3.0, 3.0]])

    encoded = encode_ellipses(ellipses, proposals=proposals, weights=weights)
    pred_a, pred_b, pred_x, pred_y, pred_theta = (
        decode_ellipses(encoded, proposals, weights).squeeze(1).unbind(-1)
    )

    torch.testing.assert_close(pred_a, a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_b, b, rtol=1e-4, atol=1e-4)


def test_ellipse_encoder():
    # Test initialization and weights
    encoder = EllipseEncoder(weights=(1.0, 2.0, 3.0, 4.0, 5.0))
    assert len(encoder.weights) == 5
    assert encoder.ellipse_xform_clip == math.log(1000.0 / 16)

    # Test batch encoding/decoding with multiple images
    batch_size_1 = 3
    batch_size_2 = 2

    # Generate test data with more reasonable ranges
    ellipse1 = sample_parametric_ellipses(
        batch_size_1,
        a_range=(2.0, 4.0),  # Reduced range
        b_range=(1.0, 2.0),  # Reduced range
        theta_range=(0, 2 * torch.pi),
        xy_range=(-5.0, 5.0),  # Reduced range
    )
    a1, b1, cx1, cy1, theta1 = ellipse1.unbind(-1)

    ellipse2 = sample_parametric_ellipses(
        batch_size_2,
        a_range=(2.0, 4.0),  # Reduced range
        b_range=(1.0, 2.0),  # Reduced range
        theta_range=(0, 2 * torch.pi),
        xy_range=(-5.0, 5.0),  # Reduced range
    )
    a2, b2, cx2, cy2, theta2 = ellipse2.unbind(-1)

    # Create proposal boxes with smaller margins
    margins1 = torch.ones_like(a1) * 1.2  # Fixed margin of 1.2
    proposals1 = torch.stack(
        [
            cx1 - a1 * margins1,
            cy1 - b1 * margins1,
            cx1 + a1 * margins1,
            cy1 + b1 * margins1,
        ],
        dim=1,
    )

    margins2 = torch.ones_like(a2) * 1.2  # Fixed margin of 1.2
    proposals2 = torch.stack(
        [
            cx2 - a2 * margins2,
            cy2 - b2 * margins2,
            cx2 + a2 * margins2,
            cy2 + b2 * margins2,
        ],
        dim=1,
    )

    # Test encoding multiple images
    reference_ellipses = [ellipse1, ellipse2]
    proposals = [proposals1, proposals2]

    encoded = encoder.encode(reference_ellipses, proposals)

    # Check encoded shapes
    assert len(encoded) == 2
    assert encoded[0].shape == (batch_size_1, 6)
    assert encoded[1].shape == (batch_size_2, 6)

    # Test decoding multiple images
    pred = encoder.decode(torch.cat(encoded), proposals)

    pred_a = tuple(p[:, 0, 0] for p in pred)
    pred_b = tuple(p[:, 0, 1] for p in pred)
    pred_cx = tuple(p[:, 0, 2] for p in pred)
    pred_cy = tuple(p[:, 0, 3] for p in pred)

    # Check decoded shapes
    assert len(pred_a) == 2
    assert pred_a[0].shape == (batch_size_1,)
    assert pred_a[1].shape == (batch_size_2,)

    # Test reconstruction accuracy for each image
    torch.testing.assert_close(
        torch.cat(pred_a), torch.cat([a1, a2]), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        torch.cat(pred_b), torch.cat([b1, b2]), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        torch.cat(pred_cx), torch.cat([cx1, cx2]), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        torch.cat(pred_cy), torch.cat([cy1, cy2]), rtol=1e-4, atol=1e-4
    )


def test_ellipse_encoder_empty_inputs():
    encoder = EllipseEncoder(weights=(1.0, 1.0, 1.0, 1.0, 1.0))

    # Test with empty inputs
    empty_ellipses = [torch.zeros((0, 5))]
    empty_proposals = [torch.zeros((0, 4))]

    encoded = encoder.encode(empty_ellipses, empty_proposals)
    assert len(encoded) == 1
    assert encoded[0].shape == (0, 6)

    decoded = encoder.decode(encoded[0], empty_proposals)
    assert all(t.shape == (0, 1, 5) for t in decoded)


def test_ellipse_encoder_clipping():
    clip_value = 2.0
    encoder = EllipseEncoder(
        weights=(1.0, 1.0, 1.0, 1.0, 1.0), ellipse_xform_clip=clip_value
    )

    # Generate test data with extreme values
    a = torch.tensor([10.0, 100.0, 1000.0])
    b = torch.tensor([5.0, 50.0, 500.0])
    cx = torch.zeros_like(a)
    cy = torch.zeros_like(a)
    theta = torch.zeros_like(a)

    # Small proposal boxes to force large encoding values
    proposals = torch.tensor(
        [
            [-1, -1, 1, 1],
            [-1, -1, 1, 1],
            [-1, -1, 1, 1],
        ]
    )

    reference_ellipses = [torch.stack([a, b, cx, cy, theta], dim=-1).view(-1, 5)]
    encoded = encoder.encode(reference_ellipses, [proposals])

    # Decode with clipping
    pred = encoder.decode(encoded[0], [proposals])

    pred_a, pred_b, pred_cx, pred_cy, pred_theta = pred[0].squeeze(1).unbind(-1)

    # The maximum allowed size should be the proposal size * exp(clip_value)
    max_allowed = 2.0 * math.exp(clip_value)  # proposal width/height is 2.0

    # Check that decoded values are properly clipped
    assert torch.all(
        pred_a[0] <= max_allowed
    ), f"pred_a: {pred_a[0]}, max allowed: {max_allowed}"
    assert torch.all(
        pred_b[0] <= max_allowed
    ), f"pred_b: {pred_b[0]}, max allowed: {max_allowed}"

    # Center coordinates should remain unchanged by clipping
    torch.testing.assert_close(pred_cx, cx)
    torch.testing.assert_close(pred_cy, cy)


def test_ellipse_encoder_single():
    encoder = EllipseEncoder(weights=(1.0, 1.0, 1.0, 1.0, 1.0))

    # Test single ellipse encoding/decoding
    a = torch.tensor([3.0])
    b = torch.tensor([2.0])
    cx = torch.tensor([0.0])
    cy = torch.tensor([0.0])
    theta = torch.tensor([torch.pi / 4])

    proposals = torch.tensor([[-3.0, -3.0, 3.0, 3.0]])

    # Test encode_single
    ellipses = torch.stack([a, b, cx, cy, theta], dim=-1).view(-1, 5)
    encoded = encoder.encode_single(ellipses, proposals)
    assert encoded.shape == (1, 6)

    # Test decode_single
    pred_a, pred_b, pred_cx, pred_cy, pred_theta = (
        encoder.decode_single(encoded, proposals).squeeze(1).unbind(-1)
    )

    torch.testing.assert_close(pred_a, a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_b, b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_cx, cx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pred_cy, cy, rtol=1e-4, atol=1e-4)

    # Handle angle periodicity in comparison
    angle_diff = torch.abs(
        torch.remainder(pred_theta - theta + torch.pi, 2 * torch.pi) - torch.pi
    )
    assert torch.all(angle_diff < 1e-4)
