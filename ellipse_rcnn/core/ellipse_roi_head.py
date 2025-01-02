from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss

from .kld import SymmetricKLDLoss
from ..utils.conics import ellipse_to_conic_matrix


class EllipseRegressor(nn.Module):
    def __init__(
        self, in_channels: int = 1024, hidden_size: int = 64, out_features: int = 3
    ):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.tanh(self.fc2(x))

        return x


def postprocess_ellipse_predictor(
    d_a: torch.Tensor, d_b: torch.Tensor, d_angle: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """Processes elliptical predictor outputs and converts them into conic matrices.

    Parameters
    ----------
    d_a : torch.Tensor
        Logarithm of scale factors for semi-major axes of ellipses.
    d_b : torch.Tensor
        Logarithm of scale factors for semi-minor axes of ellipses.
    d_angle : torch.Tensor
        Normalized rotation angles of ellipses, scaled between 0 and 1.
    boxes : torch.Tensor
        Tensor containing bounding box information, with shape (N, 4). Each box
        is represented as a 4-tuple (x_min, y_min, x_max, y_max).

    Returns
    -------
    torch.Tensor
        Conic matrix representations of predicted ellipses. Each output matrix
        corresponds to the real-valued ellipse described by the inputs.
    """
    # Pre-compute box width, height, and diagonal
    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    box_diagonal = torch.sqrt(box_width**2 + box_height**2)

    # Compute center coordinates
    center_x = boxes[:, 0] + (box_width / 2)
    center_y = boxes[:, 1] + (box_height / 2)

    # Compute ellipse parameters
    semi_axes = [(torch.exp(param) * box_diagonal / 2) for param in (d_a, d_b)]
    semi_major_axis, semi_minor_axis = semi_axes
    rotation_angles = d_angle * torch.pi

    cos_theta = torch.cos(rotation_angles)
    sin_theta = torch.sin(rotation_angles)
    rotation_angles = torch.where(
        cos_theta >= 0,
        torch.atan2(sin_theta, cos_theta),
        torch.atan2(-sin_theta, -cos_theta),
    )

    # Convert ellipse parameters to conic matrix
    return ellipse_to_conic_matrix(
        semi_major_axis, semi_minor_axis, center_x, center_y, rotation_angles
    )


def ellipse_loss_kld(
    d_pred: torch.Tensor,
    ellipse_matrix_targets: List[torch.Tensor],
    pos_matched_idxs: List[torch.Tensor],
    boxes: List[torch.Tensor],
    kld_loss: SymmetricKLDLoss,
) -> torch.Tensor:
    A_target = torch.cat(
        [o[idxs] for o, idxs in zip(ellipse_matrix_targets, pos_matched_idxs)], dim=0
    )
    boxes = torch.cat(boxes, dim=0)

    if A_target.numel() == 0:
        return d_pred.sum() * 0

    d_a = d_pred[:, 0]
    d_b = d_pred[:, 1]
    d_angle = d_pred[:, 2]

    A_pred = postprocess_ellipse_predictor(d_a, d_b, d_angle, boxes)

    loss = kld_loss.forward(A_pred, A_target).clip(0.0, 10.0).mean()

    return loss


class EllipseRoIHeads(RoIHeads):
    def __init__(
        self,
        box_roi_pool: nn.Module,
        box_head: nn.Module,
        box_predictor: nn.Module,
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        bbox_reg_weights: Optional[Tuple[float, float, float, float]],
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
        ellipse_roi_pool: nn.Module,
        ellipse_head: nn.Module,
        ellipse_predictor: nn.Module,
        # Loss parameters
        kld_shape_only: bool = True,
        kld_normalize: bool = True,
        kld_nan_to_num: float = float(1e4),
        loss_scale: float = 1.0,
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
        )

        self.ellipse_roi_pool = ellipse_roi_pool
        self.ellipse_head = ellipse_head
        self.ellipse_predictor = ellipse_predictor

        self.kld_loss = SymmetricKLDLoss(
            shape_only=kld_shape_only,
            normalize=kld_normalize,
            nan_to_num=kld_nan_to_num,
        )
        self.ellipse_loss_fn = ellipse_loss_kld
        self.loss_scale = loss_scale

    def has_ellipse_reg(self) -> bool:
        if self.ellipse_roi_pool is None:
            return False
        if self.ellipse_head is None:
            return False
        if self.ellipse_predictor is None:
            return False
        return True

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError("target boxes must be of float type")
                if t["ellipse_matrices"].dtype not in floating_point_types:
                    raise TypeError("target ellipse_offsets must be of float type")
                if t["labels"].dtype != torch.int64:
                    raise TypeError("target labels must be of int64 type")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = (
                self.select_training_samples(proposals, targets)
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None or regression_targets is None:
                raise ValueError(
                    "Labels and regression targets must not be None during training"
                )
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_ellipse_reg():
            ellipse_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("matched_idxs must not be None during training")
                # during training, only focus on positive boxes
                num_images = len(proposals)
                ellipse_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    ellipse_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None  # type: ignore

            if self.ellipse_roi_pool is not None:
                ellipse_features = self.ellipse_roi_pool(
                    features, ellipse_proposals, image_shapes
                )
                ellipse_features = self.ellipse_head(ellipse_features)
                ellipse_shapes_normalised = self.ellipse_predictor(ellipse_features)
            else:
                raise Exception("Expected ellipse_roi_pool to be not None")

            loss_ellipse_regressor = {}
            if self.training:
                if targets is None:
                    raise ValueError("Targets must not be None during training")
                if pos_matched_idxs is None:
                    raise ValueError(
                        "pos_matched_idxs must not be None during training"
                    )
                if ellipse_shapes_normalised is None:
                    raise ValueError(
                        "ellipse_shapes_normalised must not be None during training"
                    )

                ellipse_matrix_targets = [t["ellipse_matrices"] for t in targets]
                rcnn_loss_ellipse = (
                    self.ellipse_loss_fn(
                        ellipse_shapes_normalised,
                        ellipse_matrix_targets,
                        pos_matched_idxs,
                        ellipse_proposals,
                        self.kld_loss,
                    )
                    * self.loss_scale
                )
                loss_ellipse_regressor = {"loss_ellipse": rcnn_loss_ellipse}
            else:
                ellipses_per_image = [lbl.shape[0] for lbl in labels]
                for e_l, r, box in zip(
                    ellipse_shapes_normalised.split(ellipses_per_image, dim=0),
                    result,
                    ellipse_proposals,
                ):
                    d_a = e_l[:, 0]
                    d_b = e_l[:, 1]
                    d_angle = e_l[:, 2]
                    r["ellipse_matrices"] = postprocess_ellipse_predictor(
                        d_a, d_b, d_angle, box
                    )

            losses.update(loss_ellipse_regressor)

        return result, losses
