import math

import torch


def encode_ellipses(
    reference_ellipses: torch.Tensor,
    proposals: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Encode ellipse parameters relative to proposal boxes.

    Parameters
    ----------
    reference_ellipses : torch.Tensor
        Ellipse parameters with shape [N, 5] with ordering [a, b, cx, cy, theta]
    proposals : torch.Tensor
        Proposal boxes from RPN in the format (x1, y1, x2, y2) [N, 4].
    weights : torch.Tensor
        Weights for (a, b, x, y, theta) parameters, shape [5].

    Returns
    -------
    torch.Tensor
        Encoded ellipse parameters relative to proposals, shape [N, 5].
    """
    a, b, cx, cy, theta = reference_ellipses.unbind(-1)
    # Proposal box parameters [N]
    ex_widths = proposals[:, 2] - proposals[:, 0]
    ex_heights = proposals[:, 3] - proposals[:, 1]
    ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
    ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
    ex_diag = torch.sqrt(ex_widths**2 + ex_heights**2)

    # Unpack weights
    wa = weights[0]
    wb = weights[1]
    wx = weights[2]
    wy = weights[3]
    wtheta = weights[4]

    # Center offset targets normalized by proposal dimensions [N]
    targets_dx = wx * (cx - ex_ctr_x) / ex_diag
    targets_dy = wy * (cy - ex_ctr_y) / ex_diag

    # Axis lengths normalized by proposal diagonal [N]
    targets_da = wa * torch.log(2 * a / ex_diag)
    targets_db = wb * torch.log(2 * b / ex_diag)

    # Angle target - normalize to [-1,1] range [N]
    targets_dtheta = wtheta * (theta / torch.pi)

    # Stack to [N, 5]
    targets = torch.stack(
        (targets_da, targets_db, targets_dx, targets_dy, targets_dtheta), dim=1
    )
    return targets


def decode_ellipses(
    ellipse_enc: torch.Tensor, proposals: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Decode relative ellipse parameters back to absolute values.

    Parameters
    ----------
    ellipse_enc : torch.Tensor
        Encoded ellipse parameters (da, db, dx, dy, dtheta) of shape [N, 5].
    proposals : torch.Tensor
        Proposal boxes from RPN in the format (x1, y1, x2, y2) of shape [N, 4].
    weights : torch.Tensor
        Weights for (a, b, x, y, theta) parameters, shape [5].

    Returns
    -------
    torch.Tensor
        containing:
            - pred_a : torch.Tensor
                Decoded major semi-axes of the ellipses of shape [N].
            - pred_b : torch.Tensor
                Decoded minor semi-axes of the ellipses of shape [N].
            - pred_cx : torch.Tensor
                Decoded x-coordinates of the ellipse centers of shape [N].
            - pred_cy : torch.Tensor
                Decoded y-coordinates of the ellipse centers of shape [N].
            - pred_theta : torch.Tensor
                Decoded orientations (angles) of the ellipses of shape [N].
    """
    # Split encoded parameters [N]
    da = ellipse_enc[:, 0] / weights[0]
    db = ellipse_enc[:, 1] / weights[1]
    dx = ellipse_enc[:, 2] / weights[2]
    dy = ellipse_enc[:, 3] / weights[3]
    dtheta = ellipse_enc[:, 4] / weights[4]

    # Proposal box parameters [N]
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights
    diag = torch.sqrt(widths**2 + heights**2)

    # Decode center coordinates [N]
    pred_cx = dx * diag + ctr_x
    pred_cy = dy * diag + ctr_y

    # Decode semi-axes [N]
    pred_a = torch.exp(da) * diag / 2
    pred_b = torch.exp(db) * diag / 2

    # Decode angle [N]
    pred_theta = dtheta * torch.pi

    # Ensure b <= a
    pred_a, pred_b = torch.maximum(pred_a, pred_b), torch.minimum(pred_a, pred_b)

    return torch.stack([pred_a, pred_b, pred_cx, pred_cy, pred_theta], dim=-1).view(
        -1, 5
    )


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
        reference_ellipses: list[torch.Tensor],
        proposals: list[torch.Tensor],
    ) -> list[torch.Tensor]:
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
        return targets.split(ellipses_per_image, 0)

    def encode_single(
        self,
        reference_ellipses: torch.Tensor,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a set of ellipses with respect to proposal boxes.

        Parameters
        ----------
        reference_ellipses : torch.Tensor
            Ellipse parameters with shape [N, 5] with ordering [a, b, cx, cy, theta]
        proposals : torch.Tensor
            Proposal boxes [N,4] in (x1,y1,x2,y2) format
        """
        dtype = proposals.dtype
        device = proposals.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)

        targets = encode_ellipses(
            reference_ellipses=reference_ellipses, proposals=proposals, weights=weights
        )

        return targets

    def decode(
        self, rel_codes: torch.Tensor, proposals: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """
        From a set of encoded relative ellipse parameters and proposal boxes,
        decode the absolute ellipse parameters.

        Args:
            rel_codes: Encoded relative parameters [N, 5]
            proposals: List of proposal boxes tensors
        """
        assert isinstance(proposals, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)

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
            return pred_ellipses.split(boxes_per_image)

        return pred_ellipses

    def decode_single(
        self, rel_codes: torch.Tensor, proposals: torch.Tensor
    ) -> torch.Tensor:
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

        # Clamp the encoded axis parameters before decoding
        clamped_params = torch.clamp(
            rel_codes[:, :2], min=-self.ellipse_xform_clip, max=self.ellipse_xform_clip
        )
        rel_codes = torch.cat([clamped_params, rel_codes[:, 2:]], dim=1)

        # Decode ellipse parameters
        pred_ellipses = decode_ellipses(
            ellipse_enc=rel_codes, proposals=proposals, weights=weights
        )

        # Clamp the decoded axis lengths to reasonable values relative to proposals
        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        max_size = torch.maximum(widths, heights) * math.exp(self.ellipse_xform_clip)

        # pred_a = torch.minimum(pred_a, max_size)
        # pred_b = torch.minimum(pred_b, max_size)

        return pred_ellipses
