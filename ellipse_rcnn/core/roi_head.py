from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops

from .encoder import EllipseEncoder
from .kld import SymmetricKLDLoss
from ellipse_rcnn.core.ops import (
    ellipse_to_conic_matrix,
    remove_small_ellipses,
    bbox_ellipse,
)


class EllipseRCNNPredictor(nn.Module):
    """
    A neural network module designed to predict encoded parameters of
    an ellipse given input features.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 5)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        ellipse_deltas = self.bbox_pred(x)

        return scores, ellipse_deltas


class EllipseLossDict(TypedDict):
    loss_ellipse_kld: torch.Tensor
    loss_ellipse_smooth_l1: torch.Tensor


def ellipse_rcnn_loss(
    class_logits: torch.Tensor,
    ellipse_regression: torch.Tensor,
    labels: list[torch.Tensor],
    regression_targets: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the loss for Ellipse R-CNN.

    Args:
        class_logits (Tensor)
        ellipse_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        ellipse_reg_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    ellipse_regression = ellipse_regression.reshape(
        N, ellipse_regression.size(-1) // 5, 5
    )

    ellipse_reg_loss = F.smooth_l1_loss(
        ellipse_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    ellipse_reg_loss = ellipse_reg_loss / labels.numel()

    return classification_loss, ellipse_reg_loss


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
        bbox_reg_weights: tuple[float, float, float, float],
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
        # Ellipse parameters
        ellipse_roi_pool: nn.Module,
        ellipse_head: nn.Module,
        ellipse_predictor: nn.Module,
        ellipse_encoder_weights: tuple[float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        # Loss parameters
        kld_shape_only: bool = False,
        kld_normalize: bool = False,
        # Numerical stability parameters
        nan_to_num: float = 10.0,
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

        self.ellipse_encoder = EllipseEncoder(weights=ellipse_encoder_weights)
        self.ellipse_roi_pool = ellipse_roi_pool
        self.ellipse_head = ellipse_head
        self.ellipse_predictor = ellipse_predictor

        self.kld_loss_fn = SymmetricKLDLoss(
            shape_only=kld_shape_only,
            normalize=kld_normalize,
            nan_to_num=nan_to_num,
        )
        self.loss_scale = loss_scale

    @staticmethod
    def has_ellipse_reg() -> bool:
        return True

    def postprocess_ellipse_regressions(self):
        pass

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError("target boxes must be of float type")
                if t["ellipse_params"].dtype not in floating_point_types:
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

        ellipse_features = self.ellipse_roi_pool(features, proposals, image_shapes)
        ellipse_features = self.ellipse_head(ellipse_features)
        class_logits, ellipse_regression = self.ellipse_predictor(ellipse_features)

        result: list[dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None or regression_targets is None:
                raise ValueError(
                    "Labels and regression targets must not be None during training"
                )
            loss_classifier, loss_ellipse_reg = ellipse_rcnn_loss(
                class_logits, ellipse_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_ellipse_reg": loss_ellipse_reg,
            }
        else:
            ellipses, scores, labels, boxes = self.postprocess_detections(
                class_logits, ellipse_regression, proposals, image_shapes
            )
            num_images = len(ellipses)
            for i in range(num_images):
                result.append(
                    {
                        "ellipse_params": ellipses[i],
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses

    def postprocess_ellipse_predictor(
        self,
        pred: torch.Tensor,
        box_proposals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processes elliptical predictor outputs and converts them into conic matrices.

        Parameters
        ----------
        pred : torch.Tensor
            Tensor containing predicted ellipse parameters, with shape (N, 5). Each parameter
            is represented as a 5-tuple (d_a, d_b, dx, dy, dtheta).
        box_proposals : torch.Tensor
            Tensor containing proposed bounding box information, with shape (N, 4). Each box
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
        d_a, d_b, d_theta = pred

        # Pre-compute box width, height, and diagonal
        box_width = box_proposals[:, 2] - box_proposals[:, 0]
        box_height = box_proposals[:, 3] - box_proposals[:, 1]
        box_diag = torch.sqrt(box_width**2 + box_height**2)

        a = box_diag * d_a.exp()
        b = box_diag * d_b.exp()

        box_x = box_proposals[:, 0] + box_width * 0.5
        box_y = box_proposals[:, 1] + box_height * 0.5

        theta = (d_theta * 2.0 - 1.0) * (torch.pi / 2)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        theta = torch.where(
            cos_theta >= 0,
            torch.atan2(sin_theta, cos_theta),
            torch.atan2(-sin_theta, -cos_theta),
        )

        return a, b, box_x, box_y, theta

    def ellipse_loss(
        self,
        pred: torch.Tensor,
        target: list[torch.Tensor],
        pos_matched_idxs: list[torch.Tensor],
        box_proposals: list[torch.Tensor],
    ) -> EllipseLossDict:
        target = torch.cat(
            [o[idxs] for o, idxs in zip(target, pos_matched_idxs)], dim=0
        )

        box_proposals = torch.cat(box_proposals, dim=0)

        if target.numel() == 0:
            return {
                "loss_ellipse_kld": torch.tensor(
                    0.0, device=pred.device, dtype=pred.dtype
                ),
                "loss_ellipse_smooth_l1": torch.tensor(
                    0.0, device=pred.device, dtype=pred.dtype
                ),
            }

        # Encode target
        target_enc = self.ellipse_encoder.encode_single(
            target,
            proposals=box_proposals,
        )

        # Direct Smooth L1 loss
        loss_smooth_l1 = F.smooth_l1_loss(
            pred, target_enc, beta=(1 / 9), reduction="sum"
        )
        loss_smooth_l1 /= box_proposals.shape[0]
        loss_smooth_l1 = loss_smooth_l1.nan_to_num(nan=0.0).clip(max=float(1e4))

        # Decode prediction
        a, b, x, y, theta = self.ellipse_encoder.decode_single(pred, box_proposals)

        A_pred = ellipse_to_conic_matrix(a=a, b=b, theta=theta, x=x, y=y)
        a_target, b_target, cx_target, cy_target, theta_target = target.unbind(-1)
        A_target = ellipse_to_conic_matrix(
            a=a_target, b=b_target, theta=theta_target, x=cx_target, y=cy_target
        )

        loss_kld = (
            self.kld_loss_fn.forward(A_pred, A_target).clip(max=float(1e4)).mean() * 0.1
        )
        return {
            "loss_ellipse_kld": loss_kld,
            "loss_ellipse_smooth_l1": loss_smooth_l1,
        }

    def select_training_samples(
        self,
        proposals: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_ellipses = [t["ellipse_params"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_ellipses = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_ellipses_in_image = gt_ellipses[img_id]
            if gt_ellipses_in_image.numel() == 0:
                gt_ellipses_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_ellipses.append(gt_ellipses_in_image[matched_idxs[img_id]])

        regression_targets = self.ellipse_encoder.encode(matched_gt_ellipses, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
        self,
        class_logits: torch.Tensor,
        ellipse_regression: torch.Tensor,
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_ellipses = self.ellipse_encoder.decode(ellipse_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        if isinstance(pred_ellipses, torch.Tensor):
            pred_ellipses = pred_ellipses.split(boxes_per_image, 0)

        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_ellipses = []
        all_boxes = []
        all_scores = []
        all_labels = []
        for ellipses, scores, image_shape in zip(
            pred_ellipses, pred_scores_list, image_shapes
        ):
            # boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            # TODO: Ellipse predictions should be [N, C, 5]?
            # ellipses = ellipses[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            ellipses = ellipses.reshape(-1, 5)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]  # type: ignore
            ellipses, scores, labels = ellipses[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = remove_small_ellipses(ellipses, min_size=1e-2)
            ellipses, scores, labels = ellipses[keep], scores[keep], labels[keep]

            boxes = bbox_ellipse(ellipses)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            ellipses, scores, labels, boxes = (
                ellipses[keep],
                scores[keep],
                labels[keep],
                boxes[keep],
            )

            all_ellipses.append(ellipses)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_ellipses, all_scores, all_labels, all_boxes
