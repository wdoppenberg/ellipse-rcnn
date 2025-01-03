from typing import Dict, List, Tuple, Optional, TypedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss

from .kld import SymmetricKLDLoss
from ..utils.conics import ellipse_to_conic_matrix, conic_center, ellipse_axes, ellipse_angle, unimodular_matrix


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - a (torch.Tensor): Computed semi-major axis of the ellipses.
        - b (torch.Tensor): Computed semi-minor axis of the ellipses.
        - x (torch.Tensor): X-coordinates of the ellipse centers.
        - y (torch.Tensor): Y-coordinates of the ellipse centers.
        - theta (torch.Tensor): Rotation angles (in radians) for the ellipses.
    
    """
    # Pre-compute box width, height, and diagonal
    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    box_diagonal = torch.sqrt(box_width**2 + box_height**2)

    # Compute center coordinates
    x = boxes[:, 0] + (box_width / 2)
    y = boxes[:, 1] + (box_height / 2)

    # Compute ellipse parameters
    a, b = ((torch.exp(param) * box_diagonal / 2) for param in (d_a, d_b))
    theta = d_angle * torch.pi

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    theta = torch.where(
        cos_theta >= 0,
        torch.atan2(sin_theta, cos_theta),
        torch.atan2(-sin_theta, -cos_theta),
    )

    return a, b, x, y, theta


class EllipseLossDict(TypedDict):
    loss_ellipse_kld: torch.Tensor
    loss_ellipse_smooth_l1: torch.Tensor


def ellipse_loss(
    d_pred: torch.Tensor,
    A_target: List[torch.Tensor],
    pos_matched_idxs: List[torch.Tensor],
    boxes: List[torch.Tensor],
    kld_loss_fn: SymmetricKLDLoss,
) -> EllipseLossDict:
    A_target = torch.cat(
        [o[idxs] for o, idxs in zip(A_target, pos_matched_idxs)], dim=0
    )
    boxes = torch.cat(boxes, dim=0)

    if A_target.numel() == 0:
        return {
            "loss_ellipse_kld": torch.tensor(0.0, device=d_pred.device, dtype=d_pred.dtype),
            "loss_ellipse_smooth_l1": torch.tensor(0.0, device=d_pred.device, dtype=d_pred.dtype),
        }

    d_a = d_pred[:, 0]
    d_b = d_pred[:, 1]
    d_angle = d_pred[:, 2]

    a, b, x, y, theta = postprocess_ellipse_predictor(d_a, d_b, d_angle, boxes)
    
    a_target, b_target = ellipse_axes(A_target)
    x_target, y_target = conic_center(A_target)
    theta_target = ellipse_angle(A_target)

    A_pred = ellipse_to_conic_matrix(a=a, b=b, theta=theta, x=x, y=y)
    # A_pred = unimodular_matrix(A_pred)
    
    loss_kld = kld_loss_fn.forward(A_pred, A_target).nan_to_num(nan=0.).clip(max=float(1e4)).mean()
    
    loss_smooth_l1 = F.smooth_l1_loss(
        torch.stack([a, b, x, y, theta], dim=1),
        torch.stack([a_target, b_target, x_target, y_target, theta_target], dim=1)
    ).nan_to_num(nan=float(1e4))


    return {"loss_ellipse_kld": loss_kld, "loss_ellipse_smooth_l1": loss_smooth_l1}


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
        kld_shape_only: bool = False,
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
        self.loss_scale = loss_scale

    def has_ellipse_reg(self) -> bool:
        if self.ellipse_roi_pool is None:
            return False
        if self.ellipse_head is None:
            return False
        if self.ellipse_predictor is None:
            return False
        return True

    def postprocess_ellipse_regressions(
        self
    ):
        pass

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
            ellipse_box_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("matched_idxs must not be None during training")
                # during training, only focus on positive boxes
                num_images = len(proposals)
                ellipse_box_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    ellipse_box_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None  # type: ignore

            if self.ellipse_roi_pool is not None:
                ellipse_features = self.ellipse_roi_pool(
                    features, ellipse_box_proposals, image_shapes
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
                rcnn_loss_ellipse = ellipse_loss(
                        ellipse_shapes_normalised,
                        ellipse_matrix_targets,
                        pos_matched_idxs,
                        ellipse_box_proposals,
                        self.kld_loss,
                    )
                
                if self.loss_scale != 1.0:
                    rcnn_loss_ellipse["loss_ellipse_kld"] *= self.loss_scale
                    rcnn_loss_ellipse["loss_ellipse_smooth_l1"] *= self.loss_scale

                loss_ellipse_regressor.update(rcnn_loss_ellipse)
            else:
                ellipses_per_image = [lbl.shape[0] for lbl in labels]
                for e_l, r, box in zip(
                    ellipse_shapes_normalised.split(ellipses_per_image, dim=0),
                    result,
                    ellipse_box_proposals,
                ):
                    d_a = e_l[:, 0]
                    d_b = e_l[:, 1]
                    d_angle = e_l[:, 2]
                    r["ellipse_matrices"] = postprocess_ellipse_predictor(
                        d_a, d_b, d_angle, box
                    )

            losses.update(loss_ellipse_regressor)

        return result, losses
