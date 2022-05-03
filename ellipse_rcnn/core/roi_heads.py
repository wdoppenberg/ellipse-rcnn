from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss

from .metrics import gaussian_angle_distance, norm_mv_kullback_leibler_divergence
from ..utils.conics import ellipse_to_conic_matrix


class EllipseRegressor(nn.Module):
    def __init__(self, in_channels: int = 1024, hidden_size: int = 512, out_features: int = 3):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return x


# TODO: Create targets preprocessor for direct Smooth L1 loss calculation
def preprocess_ellipse_targets(A_target: torch.Tensor) -> torch.Tensor:
    pass


def postprocess_ellipse_predictor(
    d_a: torch.Tensor, d_b: torch.Tensor, d_angle: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    box_diag = torch.sqrt((boxes[:, 2] - boxes[:, 0]) ** 2 + (boxes[:, 2] - boxes[:, 0]) ** 2)
    cx = boxes[:, 0] + ((boxes[:, 2] - boxes[:, 0]) / 2)
    cy = boxes[:, 1] + ((boxes[:, 3] - boxes[:, 1]) / 2)

    a, b = ((torch.exp(param) * box_diag / 2) for param in (d_a, d_b))
    theta = d_angle * np.pi / 2
    ang_cond1 = torch.cos(theta) >= 0
    ang_cond2 = ~ang_cond1

    if not torch.onnx.is_in_onnx_export():
        theta[ang_cond1] = torch.atan2(torch.sin(theta[ang_cond1]), torch.cos(theta[ang_cond1]))
        theta[ang_cond2] = torch.atan2(-torch.sin(theta[ang_cond2]), -torch.cos(theta[ang_cond2]))

    return ellipse_to_conic_matrix(a, b, cx, cy, theta)


def ellipse_loss_KLD(
    d_pred: torch.Tensor,
    ellipse_matrix_targets: List[torch.Tensor],
    pos_matched_idxs: List[torch.Tensor],
    boxes: List[torch.Tensor],
) -> torch.Tensor:
    A_target = torch.cat([o[idxs] for o, idxs in zip(ellipse_matrix_targets, pos_matched_idxs)], dim=0)
    boxes = torch.cat(boxes, dim=0)

    if A_target.numel() == 0:
        return d_pred.sum() * 0

    d_a = d_pred[:, 0]
    d_b = d_pred[:, 1]
    d_angle = d_pred[:, 2]

    A_pred = postprocess_ellipse_predictor(d_a, d_b, d_angle, boxes)

    loss1 = norm_mv_kullback_leibler_divergence(A_pred, A_target)
    loss2 = norm_mv_kullback_leibler_divergence(A_target, A_pred)

    return (0.5 * loss1 + 0.5 * loss2).mean()


def ellipse_loss_GA(
    d_pred: torch.Tensor,
    ellipse_matrix_targets: List[torch.Tensor],
    pos_matched_idxs: List[torch.Tensor],
    boxes: List[torch.Tensor],
) -> torch.Tensor:
    A_target = torch.cat([o[idxs] for o, idxs in zip(ellipse_matrix_targets, pos_matched_idxs)], dim=0)
    boxes = torch.cat(boxes, dim=0)

    if A_target.numel() == 0:
        return d_pred.sum() * 0

    d_a = d_pred[:, 0]
    d_b = d_pred[:, 1]
    d_angle = d_pred[:, 2]

    A_pred = postprocess_ellipse_predictor(d_a, d_b, d_angle, boxes)

    return gaussian_angle_distance(A_pred, A_target).mean()


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
        ellipse_loss_metric: str = "gaussian-angle",
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

        if ellipse_loss_metric == "gaussian-angle":
            self.ellipse_loss_fn = ellipse_loss_GA
        elif ellipse_loss_metric == "kullback-leibler":
            self.ellipse_loss_fn = ellipse_loss_KLD
        else:
            raise ValueError(f"Ellipse loss function {ellipse_loss_metric} not known.")

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
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["ellipse_matrices"].dtype in floating_point_types, "target ellipse_offsets must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
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
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
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
                assert matched_idxs is not None
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
                ellipse_features = self.ellipse_roi_pool(features, ellipse_proposals, image_shapes)
                ellipse_features = self.ellipse_head(ellipse_features)
                ellipse_shapes_normalised = self.ellipse_predictor(ellipse_features)
            else:
                raise Exception("Expected ellipse_roi_pool to be not None")

            loss_ellipse_regressor = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert ellipse_shapes_normalised is not None

                ellipse_matrix_targets = [t["ellipse_matrices"] for t in targets]
                rcnn_loss_ellipse = self.ellipse_loss_fn(
                    ellipse_shapes_normalised, ellipse_matrix_targets, pos_matched_idxs, ellipse_proposals
                )
                loss_ellipse_regressor = {"loss_ellipse": rcnn_loss_ellipse}
            else:
                ellipses_per_image = [lbl.shape[0] for lbl in labels]
                for e_l, r, box in zip(
                    ellipse_shapes_normalised.split(ellipses_per_image, dim=0), result, ellipse_proposals
                ):
                    d_a = e_l[:, 0]
                    d_b = e_l[:, 1]
                    d_angle = e_l[:, 2]
                    r["ellipse_matrices"] = postprocess_ellipse_predictor(d_a, d_b, d_angle, box)

            losses.update(loss_ellipse_regressor)

        return result, losses
