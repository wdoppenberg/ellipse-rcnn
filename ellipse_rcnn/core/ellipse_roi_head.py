from typing import Dict, List, Tuple, Optional, TypedDict, NamedTuple, Self

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss

from .kld import SymmetricKLDLoss
from .wd import WassersteinLoss
from ..utils.conics import (
    ellipse_to_conic_matrix,
    ellipse_axes,
    ellipse_angle,
    conic_center,
)


class RegressorPrediction(NamedTuple):
    """
    Represents the processed outputs of a regression model as a named tuple.

    This class encapsulates regression model outputs in a structured format, where
    each attribute corresponds to a specific component of the regression output.
    These outputs can be directly used for post-processing steps such as transformation
    into conic matrices or further evaluations of ellipse geometry.

    Attributes
    ----------
    d_a : torch.Tensor
        The normalized semi-major axis scale factor (logarithmic) used to compute
        the actual semi-major axis length of ellipses.
    d_b : torch.Tensor
        The normalized semi-minor axis scale factor (logarithmic) used to compute
        the actual semi-minor axis length of ellipses.
    d_x : torch.Tensor
        The normalized x-coordinate translation factor, specifying the adjustment
        to the center of bounding boxes for ellipse placement.
    d_y : torch.Tensor
        The normalized y-coordinate translation factor, specifying the adjustment
        to the center of bounding boxes for ellipse placement.
    d_theta : torch.Tensor
        The normalized rotation angle factor which is processed to derive the
        actual rotation angle (in radians) of ellipses.

    Notes
    -----
    - The attributes `d_a` and `d_b`, representing scale factors for the semi-major
      and semi-minor axes, are typically bounded between 0 and 1 using a sigmoid activation.
    - The attributes `d_x` and `d_y` serve as adjustments to bounding box centers, normalized
      with respect to the bounding box diagonals.
    - The attribute `d_theta` is normalized to ensure the rotation angle lies within
      a valid range (after transformation, typically between -π/2 and π/2 radians).
    - These normalized outputs are post-processed together with bounding box information
      to construct actionable ellipse parameters such as their axes lengths, centers,
      and angles.
    - This structure simplifies downstream regression tasks, such as conversion into
      conic matrices or calculation of geometrical losses.
    """

    d_a: torch.Tensor
    d_b: torch.Tensor
    d_theta: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.d_a.device

    @property
    def dtype(self) -> torch.dtype:
        return self.d_a.dtype

    def split(self, split_size: list[int] | int, dim: int = 0) -> list[Self]:
        return [
            RegressorPrediction(*tensors)
            for tensors in zip(
                *[torch.split(attr, split_size, dim=dim) for attr in self]
            )
        ]


class EllipseRegressor(nn.Module):
    """
    EllipseRegressor is a neural network module designed to predict parameters of
    an ellipse given input features.

    This class is a PyTorch module that uses a feedforward neural network to predict
    the normalized five parameters of an ellipse: semi-major axis `a`, semi-minor axis `b`, center
    coordinates (`x`, `y`), and orientation `theta`. The class includes mechanisms
    for batch normalization and uses Xavier weight initialization for improved
    training stability and convergence.

    Attributes
    ----------
    ffnn : nn.Sequential
        A feedforward neural network with two hidden layers and ReLU activations.
    """

    def __init__(self, in_channels: int = 1024, hidden_size: int = 64):
        super().__init__()
        # Separate prediction heads for better gradient flow
        self.ffnn = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Tanh(),
        )

        # Initialize with small values
        for lin in self.ffnn:
            if isinstance(lin, nn.Linear):
                nn.init.xavier_uniform_(lin.weight, gain=0.01)
                nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> RegressorPrediction:
        x = x.flatten(start_dim=1)
        x = self.ffnn(x)

        d_a, d_b, d_theta = x.unbind(dim=-1)

        return RegressorPrediction(d_a=d_a, d_b=d_b, d_theta=d_theta)


def postprocess_ellipse_predictor(
    pred: RegressorPrediction,
    box_proposals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Processes elliptical predictor outputs and converts them into conic matrices.

    Parameters
    ----------
    pred : RegressorPrediction
        The output of the elliptical predictor model.
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


class EllipseLossDict(TypedDict):
    loss_ellipse_kld: torch.Tensor
    loss_ellipse_smooth_l1: torch.Tensor
    loss_ellipse_wasserstein: torch.Tensor


def ellipse_loss(
    pred: RegressorPrediction,
    A_target: List[torch.Tensor],
    pos_matched_idxs: List[torch.Tensor],
    box_proposals: List[torch.Tensor],
    kld_loss_fn: SymmetricKLDLoss,
    wd_loss_fn: WassersteinLoss,
) -> EllipseLossDict:
    pos_matched_idxs_batched = torch.cat(pos_matched_idxs, dim=0)
    A_target = torch.cat(A_target, dim=0)[pos_matched_idxs_batched]

    box_proposals = torch.cat(box_proposals, dim=0)

    if A_target.numel() == 0:
        return {
            "loss_ellipse_kld": torch.tensor(0.0, device=pred.device, dtype=pred.dtype),
            "loss_ellipse_smooth_l1": torch.tensor(
                0.0, device=pred.device, dtype=pred.dtype
            ),
            "loss_ellipse_wasserstein": torch.tensor(
                0.0, device=pred.device, dtype=pred.dtype
            ),
        }

    a_target, b_target = ellipse_axes(A_target)
    theta_target = ellipse_angle(A_target)

    # Box proposal parameters
    box_width = box_proposals[:, 2] - box_proposals[:, 0]
    box_height = box_proposals[:, 3] - box_proposals[:, 1]
    box_diag = torch.sqrt(box_width**2 + box_height**2).clamp(min=1e-6)

    # Normalize target variables
    da_target = (a_target / box_diag).log()
    db_target = (b_target / box_diag).log()
    dtheta_target = (theta_target / (torch.pi / 2) + 1) / 2

    # Direct parameter losses
    d_a, d_b, d_theta = pred

    pred_t = torch.stack([d_a, d_b, d_theta], dim=1)
    target_t = torch.stack([da_target, db_target, dtheta_target], dim=1)

    loss_smooth_l1 = F.smooth_l1_loss(pred_t, target_t, beta=(1 / 9), reduction="sum")
    loss_smooth_l1 /= box_proposals.shape[0]
    loss_smooth_l1 = loss_smooth_l1.nan_to_num(nan=0.0).clip(max=float(1e4))

    a, b, x, y, theta = postprocess_ellipse_predictor(pred, box_proposals)

    A_pred = ellipse_to_conic_matrix(a=a, b=b, theta=theta, x=x, y=y)

    loss_kld = kld_loss_fn.forward(A_pred, A_target).clip(max=float(1e4)).mean() * 0.1
    loss_wd = torch.zeros(1, device=pred.device, dtype=pred.dtype)
    # loss_wd = wd_loss_fn.forward(A_pred, A_target).clip(max=float(1e4)).mean() * 0.1

    return {
        "loss_ellipse_kld": loss_kld,
        "loss_ellipse_smooth_l1": loss_smooth_l1,
        "loss_ellipse_wasserstein": loss_wd,
    }


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

        self.ellipse_roi_pool = ellipse_roi_pool
        self.ellipse_head = ellipse_head
        self.ellipse_predictor = ellipse_predictor

        self.kld_loss = SymmetricKLDLoss(
            shape_only=kld_shape_only,
            normalize=kld_normalize,
            nan_to_num=nan_to_num,
        )
        self.wd_loss = WassersteinLoss(
            nan_to_num=nan_to_num,
            normalize=kld_normalize,
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

    def postprocess_ellipse_regressions(self):
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
                    self.wd_loss,
                )

                if self.loss_scale != 1.0:
                    rcnn_loss_ellipse["loss_ellipse_kld"] *= self.loss_scale
                    rcnn_loss_ellipse["loss_ellipse_smooth_l1"] *= self.loss_scale

                loss_ellipse_regressor.update(rcnn_loss_ellipse)
            else:
                ellipses_per_image = [lbl.shape[0] for lbl in labels]
                for pred, r, box in zip(
                    ellipse_shapes_normalised.split(ellipses_per_image, dim=0),
                    result,
                    ellipse_box_proposals,
                ):
                    a, b, x, y, theta = postprocess_ellipse_predictor(pred, box)
                    A_pred = ellipse_to_conic_matrix(a=a, b=b, theta=theta, x=x, y=y)
                    r["ellipse_matrices"] = A_pred
                    # r["boxes"] = bbox_ellipse(A_pred)

            losses.update(loss_ellipse_regressor)

        return result, losses
