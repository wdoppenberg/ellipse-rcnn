from abc import ABC, abstractmethod
from typing import Tuple

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from .types import TargetDict, ImageTargetTuple, CollatedBatchType, UncollatedBatchType
from .conics import bbox_ellipse


def collate_fn(batch: UncollatedBatchType) -> CollatedBatchType:
    """
    Collate function for the :class:`DataLoader`.

    Parameters
    ----------
    batch:
        A batch of data.
    """
    return tuple(zip(*batch))  # type: ignore


class EllipseDatasetBase(ABC, Dataset):
    def __init__(
        self,
        data_file: str,
        transform: torch.nn.Module,
    ) -> None:
        self.data_file = data_file
        self.transform = transform

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, TargetDict]:
        pass


class CraterEllipseDataset(Dataset):
    """
    Dataset for crater ellipse detection. Mostly meant as an example in combination with
    https://github.com/wdoppenberg/crater-detection.
    """

    def __init__(self, file_path: str, group: str):
        self.file_path = file_path
        self.group = group

    def __getitem__(self, idx: torch.Tensor) -> ImageTargetTuple:
        with h5py.File(self.file_path, "r") as dataset:
            image = torch.tensor(dataset[self.group]["images"][idx])
            start_idx = dataset[self.group]["craters/crater_list_idx"][idx]
            end_idx = dataset[self.group]["craters/crater_list_idx"][idx + 1]
            A_craters = torch.tensor(dataset[self.group]["craters/A_craters"][start_idx:end_idx])

        boxes = bbox_ellipse(A_craters)
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
            ellipse_matrices=A_craters,
        )

        return image, target

    def __len__(self) -> int:
        with h5py.File(self.file_path, "r") as f:
            return len(f[self.group]["images"])


def get_dataloaders(
    dataset_path: str, batch_size: int = 32, num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = CraterEllipseDataset(file_path=dataset_path, group="training")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=True
    )

    validation_dataset = CraterEllipseDataset(file_path=dataset_path, group="validation")
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=True
    )

    test_dataset = CraterEllipseDataset(file_path=dataset_path, group="test")
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=True)

    return train_loader, validation_loader, test_loader
