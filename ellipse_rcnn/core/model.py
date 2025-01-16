from types import NoneType
from typing import List, Tuple, Optional

from torch import nn, Tensor
from torchvision.models import ResNet50_Weights, WeightsEnum
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import TwoMLPHead  # noqa: F
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign

from .roi_head import EllipseRoIHeads, EllipseRCNNPredictor
from ellipse_rcnn.core.types import (
    TargetDict,
    LossDict,
    PredictionDict,
)
from .transform import EllipseRCNNTransform


class EllipseRCNN(GeneralizedRCNN):
    def __init__(
        self,
        num_classes: int = 2,
        # transform parameters
        backbone_name: str = "resnet50",
        weights: WeightsEnum | str = ResNet50_Weights.IMAGENET1K_V1,
        min_size: int = 256,
        max_size: int = 512,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # Region Proposal Network parameters
        rpn_anchor_generator: Optional[nn.Module] = None,
        rpn_head: Optional[nn.Module] = None,
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        rpn_score_thresh: float = 0.0,
        # Box parameters
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        bbox_reg_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        # Ellipse regressor
        ellipse_roi_pool: Optional[nn.Module] = None,
        ellipse_head: Optional[nn.Module] = None,
        ellipse_predictor: Optional[nn.Module] = None,
        ellipse_loss_scale: float = 1.0,
        ellipse_loss_normalize: bool = False,
    ):
        if backbone_name != "resnet50" and weights == ResNet50_Weights.IMAGENET1K_V1:
            raise ValueError(
                "If backbone_name is not resnet50, weights_enum must be specified"
            )

        backbone = resnet_fpn_backbone(
            backbone_name=backbone_name, weights=weights, trainable_layers=5
        )

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, NoneType)):
            raise TypeError(
                "rpn_anchor_generator must be an instance of AnchorGenerator or None"
            )

        if not isinstance(ellipse_roi_pool, (MultiScaleRoIAlign, NoneType)):
            raise TypeError(
                "box_roi_pool must be an instance of MultiScaleRoIAlign or None"
            )

        if num_classes is not None:
            if ellipse_predictor is not None:
                raise ValueError(
                    "num_classes should be None when box_predictor is specified"
                )
        else:
            if ellipse_predictor is None:
                raise ValueError(
                    "num_classes should not be None when box_predictor "
                    "is not specified"
                )

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test
        )

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        default_representation_size = 1024

        if ellipse_roi_pool is None:
            ellipse_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )

        resolution = ellipse_roi_pool.output_size[0]
        if ellipse_head is None:
            if isinstance(resolution, int):
                ellipse_head = TwoMLPHead(
                    out_channels * resolution**2, default_representation_size
                )
            else:
                raise ValueError(
                    "resolution should be an int but is {}".format(resolution)
                )

        if ellipse_predictor is None:
            ellipse_predictor = EllipseRCNNPredictor(
                default_representation_size, num_classes
            )

        roi_heads = EllipseRoIHeads(
            # Box
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            # Ellipse
            ellipse_roi_pool=ellipse_roi_pool,
            ellipse_head=ellipse_head,
            ellipse_predictor=ellipse_predictor,
            loss_scale=ellipse_loss_scale,
            kld_normalize=ellipse_loss_normalize,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = EllipseRCNNTransform(min_size, max_size, image_mean, image_std)

        super().__init__(backbone, rpn, roi_heads, transform)

    def forward(
        self, images: list[Tensor], targets: list[TargetDict] | None = None
    ) -> LossDict | list[PredictionDict]:
        return super().forward(images, targets)  # type: ignore
