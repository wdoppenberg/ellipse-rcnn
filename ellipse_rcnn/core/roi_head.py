from typing import TypedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models.detection._utils import (
    BoxCoder,
    BalancedPositiveNegativeSampler,
    Matcher,
)  # noqa: F
from torchvision.ops import boxes as box_ops

from ellipse_rcnn.core.ops import (
    remove_small_ellipses,
    bbox_ellipse,
)
from .encoder import EllipseEncoder
from .kld import SymmetricKLDLoss
from .types import TargetDict


class RoILossDict(TypedDict, total=False):
    loss_classifier: Tensor
    loss_ellipse_reg: Tensor


class RoIPredictionDict(TypedDict, total=False):
    ellipse_params: Tensor
    boxes: Tensor
    labels: Tensor
    scores: Tensor


class EllipseRCNNPredictor(nn.Module):
    """
    A neural network module designed to predict encoded parameters of
    an ellipse given input features.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)

        self.ellipse_pred = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, num_classes * 5),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the classification logits and normalized ellipse parameters.

        Parameters
        ----------
        x : Tensor
            Input tensor representing the feature map from the previous layers.

        Returns
        -------
        tuple[Tensor, Tensor]
            A tuple containing:
                * classification logits
                * normalized ellipse parameters [da, db, dcx, dcy, dsin, dcos]
        """
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)

        # Get raw predictions
        ellipse_deltas = self.ellipse_pred(x)

        # Split into components
        num_preds = ellipse_deltas.shape[0]
        ellipse_deltas = ellipse_deltas.reshape(num_preds, -1, 5)

        # Extract angle predictions and convert to sin/cos
        da = ellipse_deltas[..., 0]
        db = ellipse_deltas[..., 1]
        dcx = ellipse_deltas[..., 2]
        dcy = ellipse_deltas[..., 3]
        dtheta = ellipse_deltas[..., 4]

        # Convert angle to normalized sin/cos
        dsin = torch.sin(2 * dtheta)
        dcos = torch.cos(2 * dtheta)

        # Recombine
        ellipse_deltas = torch.stack([da, db, dcx, dcy, dsin, dcos], dim=-1).reshape(
            num_preds, -1
        )

        return scores, ellipse_deltas


def ellipse_rcnn_loss(
    class_logits: Tensor,
    ellipse_regression: Tensor,
    labels_cat: list[Tensor],
    regression_targets: list[Tensor],
) -> RoILossDict:
    """
    Computes the loss for Ellipse R-CNN.

    Parameters
    ----------
    class_logits : Tensor
        The predicted class logits for each proposal.
    ellipse_regression : Tensor
        The predicted ellipse parameters for each proposal.
    labels_cat : list[Tensor]
        The ground truth class labels for each proposal.
    regression_targets : list[Tensor]
        The ground truth regression targets for ellipse parameters.

    Returns
    -------
    RoILossDict
        A dictionary containing the following loss components:
        - loss_classifier : Tensor
            The classification loss.
        - loss_ellipse_reg : Tensor
            The regression loss for ellipse parameters.
    """
    labels_cat = torch.cat(labels_cat, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels_cat)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]  # type: ignore
    labels_pos = labels_cat[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    ellipse_regression = ellipse_regression.reshape(
        N, ellipse_regression.size(-1) // 6, 6
    )

    ellipse_reg_loss = F.smooth_l1_loss(
        ellipse_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    ellipse_reg_loss = ellipse_reg_loss / labels_cat.numel()  # type: ignore

    return RoILossDict(
        loss_classifier=classification_loss,
        loss_ellipse_reg=ellipse_reg_loss,
    )


class EllipseRoIHeads(nn.Module):
    def __init__(
        self,
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
        super().__init__()
        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(
            fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = BoxCoder(bbox_reg_weights)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

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

    def check_targets(self, targets: list[TargetDict] | None) -> None:
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")
        if not all(["ellipse_params" in t for t in targets]):
            raise ValueError(
                "Every element of targets should have a ellipse_params key"
            )

    def assign_targets_to_proposals(
        self, proposals: list[Tensor], gt_boxes: list[Tensor], gt_labels: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
            proposals, gt_boxes, gt_labels
        ):
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_image, proposals_in_image
                )
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = (
                    matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                )
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = (
                    matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                )
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels: list[Tensor]) -> list[Tensor]:
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    @staticmethod
    def add_gt_proposals(
        proposals: list[Tensor], gt_boxes: list[Tensor]
    ) -> list[Tensor]:
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[TargetDict] | None = None,
    ) -> tuple[list[RoIPredictionDict], RoILossDict]:
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

        pred: list[RoIPredictionDict] = []
        losses: RoILossDict = {}

        if self.training:
            if labels is None or regression_targets is None:
                raise ValueError(
                    "Labels and regression targets must not be None during training"
                )
            losses = ellipse_rcnn_loss(
                class_logits, ellipse_regression, labels, regression_targets
            )
        else:
            pred = []
            ellipses, scores, labels, boxes = self.postprocess_detections(
                class_logits, ellipse_regression, proposals, image_shapes
            )
            num_images = len(ellipses)
            for i in range(num_images):
                pred.append(
                    {
                        "ellipse_params": ellipses[i],
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return pred, losses

    def select_training_samples(
        self,
        proposals: list[Tensor],
        targets: list[TargetDict] | None,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_ellipses = [t["ellipse_params"] for t in targets]

        # Append ground-truth bboxes to proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # Get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )

        # Sample a fixed proportion of positive-negative proposals
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
                gt_ellipses_in_image = torch.zeros((1, 5), dtype=dtype, device=device)
            matched_gt_ellipses.append(gt_ellipses_in_image[matched_idxs[img_id]])

        regression_targets = self.ellipse_encoder.encode(matched_gt_ellipses, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
        self,
        class_logits: Tensor,
        ellipse_regression: Tensor,
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_ellipses = self.ellipse_encoder.decode(ellipse_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        if isinstance(pred_ellipses, Tensor):
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
            labels = labels.reshape(1, -1).expand_as(scores)

            # remove predictions with the background label
            # TODO: Ellipse predictions should be [N, C, 5]?
            ellipses = ellipses[:, 1:]
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
            keep = remove_small_ellipses(ellipses, min_size=1e-1)
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
