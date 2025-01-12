import math

import torch
from torch import Tensor


def encode_ellipses(
    reference_ellipses: Tensor,
    proposals: Tensor,
    weights: Tensor,
) -> Tensor:
    """
    Encode ellipse parameters relative to proposal boxes.

    Parameters
    ----------
    reference_ellipses : Tensor
        Ellipse parameters with shape [N, 5] with ordering [a, b, cx, cy, theta]
    proposals : Tensor
        Proposal boxes from RPN in the format (x1, y1, x2, y2) [N, 4].
    weights : Tensor
        Weights for (a, b, x, y, theta) parameters, shape [5].

    Returns
    -------
    Tensor
        Encoded ellipse parameters relative to proposals, shape [N, 5].
    """
    a, b, cx, cy, theta = reference_ellipses.unbind(-1)

    # Proposal box parameters [N]
    ex_widths = proposals[:, 2] - proposals[:, 0]
    ex_heights = proposals[:, 3] - proposals[:, 1]
    ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
    ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
    ex_diag = torch.sqrt(ex_widths**2 + ex_heights**2)

    # Get the normalized angle by checking if axes need to be swapped
    swap_mask = a < b
    theta_normalized = torch.where(swap_mask, theta + torch.pi / 2, theta)
    theta_normalized = torch.atan2(
        torch.sin(theta_normalized), torch.cos(theta_normalized)
    )

    # Encode using sin and cos instead of raw angle for continuity
    wt = weights[4]
    targets_sin = wt * torch.sin(2 * theta_normalized)
    targets_cos = wt * torch.cos(2 * theta_normalized)

    # Center offset and axis length targets as before
    targets_da = weights[0] * torch.log(2 * a / ex_diag)
    targets_db = weights[1] * torch.log(2 * b / ex_diag)
    targets_dx = weights[2] * (cx - ex_ctr_x) / ex_diag
    targets_dy = weights[3] * (cy - ex_ctr_y) / ex_diag

    return torch.stack(
        (targets_da, targets_db, targets_dx, targets_dy, targets_sin, targets_cos),
        dim=1,
    )


def decode_ellipses(ellipse_enc: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Decode relative ellipse parameters back to absolute values.

    Parameters
    ----------
    ellipse_enc : Tensor
        Encoded ellipse parameters (da, db, dx, dy, dtheta) of shape [N, 5].
    proposals : Tensor
        Proposal boxes from RPN in the format (x1, y1, x2, y2) of shape [N, 4].
    weights : Tensor
        Weights for (a, b, x, y, theta) parameters, shape [5].

    Returns
    -------
    Tensor
        containing:
            - pred_a : Tensor
                Decoded major semi-axes of the ellipses of shape [N].
            - pred_b : Tensor
                Decoded minor semi-axes of the ellipses of shape [N].
            - pred_cx : Tensor
                Decoded x-coordinates of the ellipse centers of shape [N].
            - pred_cy : Tensor
                Decoded y-coordinates of the ellipse centers of shape [N].
            - pred_theta : Tensor
                Decoded orientations (angles) of the ellipses of shape [N].
    """
    # Split encoded parameters [N]
    da = ellipse_enc[:, 0::6] / weights[0]
    db = ellipse_enc[:, 1::6] / weights[1]
    dx = ellipse_enc[:, 2::6] / weights[2]
    dy = ellipse_enc[:, 3::6] / weights[3]
    dsin = ellipse_enc[:, 4::6] / weights[4]
    dcos = ellipse_enc[:, 5::6] / weights[4]

    # Proposal box parameters [N]
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = (proposals[:, 0] + 0.5 * widths).unsqueeze(1)
    ctr_y = (proposals[:, 1] + 0.5 * heights).unsqueeze(1)
    diag = torch.sqrt(widths**2 + heights**2).unsqueeze(1)

    # Decode center coordinates [N]
    pred_cx = dx * diag + ctr_x
    pred_cy = dy * diag + ctr_y

    # Decode semi-axes [N]
    pred_a = torch.exp(da) * (diag / 2)
    pred_b = torch.exp(db) * (diag / 2)

    # Decode angle using arctan2
    pred_theta = 0.5 * torch.atan2(dsin, dcos)

    # Handle axis swapping without inplace operations
    swap_mask = pred_b > pred_a
    final_a = torch.where(swap_mask, pred_b, pred_a)
    final_b = torch.where(swap_mask, pred_a, pred_b)
    final_theta = torch.where(swap_mask, pred_theta + torch.pi / 2, pred_theta)

    # Normalize angle to [-π/2, π/2]
    final_theta = torch.atan2(torch.sin(final_theta), torch.cos(final_theta))

    return torch.stack([final_a, final_b, pred_cx, pred_cy, final_theta], dim=-1)


class EllipseEncoder:
    """
    This class encodes and decodes a set of ellipses into
    the representation used for training the regressors.
    """

    def __init__(
        self,
        weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
        ellipse_xform_clip: float = math.log(1000.0 / 16),
    ) -> None:
        """
        Args:
            weights (5-element tuple): Weights for (a, b, x, y, theta)
            ellipse_xform_clip (float): Clip value for axis transformations
        """
        self.weights = weights
        self.ellipse_xform_clip = ellipse_xform_clip

    def encode(
        self,
        reference_ellipses: list[Tensor],
        proposals: list[Tensor],
    ) -> list[Tensor]:
        """
        Encode a list of reference ellipses with respect to proposal boxes.

        Args:
            reference_ellipses: List of tuples containing (a, b, cx, cy, theta) tensors
            proposals: List of proposal boxes tensors [N, 4] in (x1,y1,x2,y2) format
        """
        # Count ellipses per image
        ellipses_per_image = [e.shape[0] for e in reference_ellipses]

        # Concatenate ellipses
        cat_ellipses = torch.cat(reference_ellipses, dim=0)

        # Concatenate proposals
        cat_proposals = torch.cat(proposals, dim=0)

        # Encode all ellipses
        targets = self.encode_single(cat_ellipses, cat_proposals)

        # Split back to per-image tensors
        targets = targets.split(ellipses_per_image, 0)
        return targets  # type: ignore

    def encode_single(
        self,
        reference_ellipses: Tensor,
        proposals: Tensor,
    ) -> Tensor:
        """
        Encode a set of ellipses with respect to proposal boxes.

        Parameters
        ----------
        reference_ellipses : Tensor
            Ellipse parameters with shape [N, 5] with ordering [a, b, cx, cy, theta]
        proposals : Tensor
            Proposal boxes [N,4] in (x1,y1,x2,y2) format
        """
        dtype = proposals.dtype
        device = proposals.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)

        targets = encode_ellipses(
            reference_ellipses=reference_ellipses, proposals=proposals, weights=weights
        )

        return targets

    def decode(self, rel_codes: Tensor, proposals: list[Tensor]) -> tuple[Tensor, ...]:
        """
        From a set of encoded relative ellipse parameters and proposal boxes,
        decode the absolute ellipse parameters.

        Args:
            rel_codes: Encoded relative parameters [N, 5]
            proposals: List of proposal boxes tensors
        """
        if not isinstance(proposals, (list, tuple)):
            raise TypeError("proposals must be a list or tuple")
        if not isinstance(rel_codes, Tensor):
            raise TypeError("rel_codes must be a Tensor")

        # Concatenate proposal boxes
        boxes_per_image = [b.size(0) for b in proposals]
        cat_boxes = torch.cat(proposals, dim=0)

        box_sum = sum(boxes_per_image)
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)

        # Decode parameters
        pred_ellipses = self.decode_single(rel_codes, cat_boxes)

        # Split parameters if needed
        if box_sum > 0:
            return pred_ellipses.split(boxes_per_image)  # type: ignore
        else:
            return (pred_ellipses,)

    def decode_single(self, rel_codes: Tensor, proposals: Tensor) -> Tensor:
        """
        Decode a set of relative ellipse parameters with respect to proposal boxes.

        Parameters
        ----------
        rel_codes:
            Encoded ellipse parameters [N, 5]
        proposals:
            Proposal boxes [N, 4] in (x1,y1,x2,y2) format
        """
        weights = torch.as_tensor(
            self.weights, dtype=rel_codes.dtype, device=rel_codes.device
        )

        # Decode ellipse parameters
        pred_ellipses = decode_ellipses(
            ellipse_enc=rel_codes, proposals=proposals, weights=weights
        )

        return pred_ellipses
