from typing import Tuple, Dict, Union

import numpy as np
import numpy.linalg as LA
import torch
from torchvision.ops import box_iou

from src.common.conics import conic_center, scale_det, ellipse_axes


def get_matched_idxs(pred: Union[Dict, torch.Tensor], target: Union[Dict, torch.Tensor], iou_threshold: float = 0.5,
                     return_iou: bool = False) -> Tuple:
    """
    Returns indices at which IoU is maximum, as well as a mask containing whether it's above iou_threshold.

    Parameters
    ----------
    pred
        Prediction boxes or dictionary
    target
        Target bounding boxes or dictionary
    iou_threshold
        Minimum IoU to consider a prediction a True Positive
    return_iou
        Whether to return IoU values of matched detections

    Returns
    -------
    Matching indices, boolean match mask

    Examples
    --------
    >>> matched_idxs, matched, iou_list = get_matched_idxs(boxes_pred, boxes_target, return_iou=True)
    >>> pred_true = pred[matched]
    >>> target_matched = target[matched_idxs][matched]

    """
    if isinstance(pred, dict) and isinstance(target, dict):
        boxes_pred = pred["boxes"]
        boxes_target = target["boxes"]
    elif isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        boxes_pred = pred
        boxes_target = target
    else:
        raise TypeError("Input must be dictionary or tensor containing bounding boxes!")

    if len(boxes_pred) > 0 and len(boxes_target) > 0:
        iou_matrix = box_iou(boxes1=boxes_pred, boxes2=boxes_target)

        iou_list, matched_idxs = iou_matrix.max(1)

        if return_iou:
            return matched_idxs, iou_list > iou_threshold, iou_list[iou_list > iou_threshold]
        else:
            return matched_idxs, iou_list > iou_threshold
    else:
        if return_iou:
            return torch.zeros(0).to(pred), torch.zeros(0, dtype=torch.bool).to(pred), torch.zeros(0).to(pred)
        else:
            return torch.zeros(0).to(pred), torch.zeros(0, dtype=torch.bool).to(pred)


def detection_metrics(pred_dict: Dict, target_dict: Dict, iou_threshold: float = 0.5,
                      confidence_threshold: float = 0.75,
                      distance_threshold: float = None) -> Tuple[float, float, float, float, float]:
    """
    Calculates Precision, Recall, F1, IoU, and Gaussian Angle Distance for a single image.

    Parameters
    ----------
    pred_dict
        Prediction output dictionary (from EllipseRCNN)
    target_dict
        Target dictionary
    iou_threshold
        Minimum IoU to consider prediction and target to be matched
    confidence_threshold
        Minimum prediction class score
    distance_threshold
        (default: None) If given, use to add another condition for determining TP, FP, FN

    Returns
    -------
    Precision, Recall, F1, IoU, Gaussian Angle Distance
    """
    scores = pred_dict["scores"]

    boxes_pred = pred_dict["boxes"]
    boxes_pred_conf = boxes_pred[scores >= confidence_threshold]
    boxes_target = target_dict["boxes"]

    if len(boxes_pred_conf) > 0 and len(boxes_target) > 0:
        A_pred = pred_dict["ellipse_matrices"]
        A_pred_conf = A_pred[scores >= confidence_threshold]
        A_target = target_dict["ellipse_matrices"]

        matched_idxs, matched, iou_list = get_matched_idxs(boxes_pred_conf, boxes_target, iou_threshold=iou_threshold,
                                                           return_iou=True)

        if len(A_target[matched_idxs][matched]) > 0:
            dist = gaussian_angle_distance(A_pred_conf, A_target[matched_idxs]).mean().item()
            iou = iou_list.mean().item()
        else:
            dist = 0.
            iou = 0.

        if distance_threshold is not None:
            matched = matched & (gaussian_angle_distance(A_pred_conf, A_target[matched_idxs]) <= distance_threshold)

        A_pred_true = A_pred_conf[matched]
        A_matched = A_target[matched_idxs][matched]

        TP = len(A_pred_true)
        FP = len(A_pred_conf) - TP

        matched_idxs_fn, matched_FN = get_matched_idxs(boxes_pred[scores < confidence_threshold],
                                                       boxes_target, iou_threshold=iou_threshold)

        if distance_threshold is not None:
            if len(matched_idxs_fn) > 0 and matched_FN.sum() > 0:
                matched_FN = matched_FN & (
                        gaussian_angle_distance(A_pred[scores < confidence_threshold],
                                                A_target[matched_idxs_fn]) <= distance_threshold)
            else:
                matched_FN = None
        else:
            matched_FN = None

        FN = len(matched_idxs_fn[matched_FN])

        precision, recall = precision_recall(TP, FP, FN)
        f1 = f1_score(precision, recall)

        return precision, recall, f1, iou, dist
    else:
        return 0., 0., 0., 0., 0.


def precision_recall(TP: int, FP: int, FN: int) -> Tuple[float, float]:
    """
    Calculates Precision and Recall from detections according to:

    .. math:: P = TP / (TP + FP)
    .. math:: R = TP / (TP + FN)

    Parameters
    ----------
    TP
        True Positives
    FP
        False Positives
    FN
        False Negatives

    Returns
    -------
    Precision, Recall

    """
    try:
        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        return precision, recall
    except ZeroDivisionError as err:
        return 0., 0.


def f1_score(precision: float, recall: float) -> float:
    """
    Calculates F1 score according to:

    .. math:: 2 (P*R)/(P+R)

    Parameters
    ----------
    precision
        Precision
    recall
        Recall

    Returns
    -------
    F1 score

    """
    try:
        return (precision * recall) / ((precision + recall) / 2)
    except ZeroDivisionError as err:
        return 0.


def mv_kullback_leibler_divergence(A1: torch.Tensor, A2: torch.Tensor, shape_only: bool = False) -> torch.Tensor:
    A1, A2 = map(scale_det, (A1, A2))
    cov1, cov2 = map(lambda arr: -arr[..., :2, :2], (A1, A2))
    m1, m2 = map(lambda arr: torch.vstack(tuple(conic_center(arr).T)).T[..., None], (A1, A2))

    trace_term = (torch.inverse(cov1) @ cov2).diagonal(dim2=-2, dim1=-1).sum(1)
    log_term = torch.log(torch.det(cov1) / torch.det(cov2))

    if shape_only:
        displacement_term = 0
    else:
        displacement_term = ((m1 - m2).transpose(-1, -2) @ cov1.inverse() @ (m1 - m2)).squeeze()

    return 0.5 * (trace_term + displacement_term - 2 + log_term)


def norm_mv_kullback_leibler_divergence(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    return 1 - torch.exp(-mv_kullback_leibler_divergence(A1, A2))


def gaussian_angle_distance(A1: Union[torch.Tensor, np.ndarray], A2: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    A1, A2 = map(scale_det, (A1, A2))
    cov1, cov2 = map(lambda arr: -arr[..., :2, :2], (A1, A2))

    if isinstance(cov1, torch.Tensor) and isinstance(cov2, torch.Tensor):
        m1, m2 = map(lambda arr: torch.vstack(tuple(conic_center(arr).T)).T[..., None], (A1, A2))

        frac_term = (4 * torch.sqrt(cov1.det() * cov2.det())) / (cov1 + cov2).det()
        exp_term = torch.exp(
            -0.5 * (m1 - m2).transpose(-1, -2) @ cov1 @ (cov1 + cov2).inverse() @ cov2 @ (m1 - m2)
        ).squeeze()

        return (frac_term * exp_term).arccos()

    elif isinstance(cov1, np.ndarray) and isinstance(cov2, np.ndarray):
        m1, m2 = map(lambda arr: np.vstack(tuple(conic_center(arr).T)).T[..., None], (A1, A2))

        frac_term = (4 * np.sqrt(LA.det(cov1) * LA.det(cov2)) / (LA.det(cov1 + cov2)))
        exp_term = np.exp(-0.5 * (m1 - m2).transpose(0, 2, 1) @ cov1 @ LA.inv(cov1 + cov2) @ cov2 @ (m1 - m2)).squeeze()

        return np.arccos(frac_term * exp_term)
    else:
        raise TypeError("A1 and A2 should of type torch.Tensor or np.ndarray.")
