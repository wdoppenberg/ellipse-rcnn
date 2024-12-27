import h5py
import torch
from torch.utils.data import Dataset

from ellipse_rcnn.utils.types import TargetDict, ImageTargetTuple
from ellipse_rcnn.utils.conics import bbox_ellipse


class CraterEllipseDataset(Dataset):
    """
    Dataset for crater ellipse detection. Mostly meant as an example in combination with
    https://github.com/wdoppenberg/crater-detection.
    """

    def __init__(self, file_path: str, group: str) -> None:
        self.file_path = file_path
        self.group = group

    def __getitem__(self, idx: torch.Tensor) -> ImageTargetTuple:
        with h5py.File(self.file_path, "r") as dataset:
            image = torch.tensor(dataset[self.group]["images"][idx])

            # The number of instances can vary, hence it was decided to use a separate array with the indices of the
            # instances.
            start_idx = dataset[self.group]["craters/crater_list_idx"][idx]
            end_idx = dataset[self.group]["craters/crater_list_idx"][idx + 1]
            ellipse_matrices = torch.tensor(
                dataset[self.group]["craters/A_craters"][start_idx:end_idx]
            )

        boxes = bbox_ellipse(ellipse_matrices)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        num_objs = len(boxes)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = TargetDict(
            boxes=boxes,
            labels=labels,
            image_id=image_id,
            area=area,
            iscrowd=iscrowd,
            ellipse_matrices=ellipse_matrices,
        )

        return image, target

    def __len__(self) -> int:
        with h5py.File(self.file_path, "r") as f:
            return len(f[self.group]["images"])
