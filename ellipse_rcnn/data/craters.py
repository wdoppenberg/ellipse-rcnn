from typing import Literal

import h5py
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ellipse_rcnn.core.types import ImageTargetTuple, TargetDict
from ellipse_rcnn.core.ops import (
    ellipse_axes,
    ellipse_center,
    ellipse_angle,
    bbox_ellipse,
)
from ellipse_rcnn.data.utils import collate_fn


type Group = Literal["training", "test", "validation"]


class CraterEllipseDataset(Dataset):
    """
    Dataset for crater ellipse detection. See https://github.com/wdoppenberg/crater-detection.
    """

    def __init__(self, file_path: str, group: Group) -> None:
        self.file_path = file_path
        self.group = group

    def __getitem__(self, idx: torch.Tensor) -> ImageTargetTuple:
        with h5py.File(self.file_path, "r") as dataset:
            image = torch.tensor(dataset[self.group]["images"][idx])

            # The number of instances can vary, therefore we use a separate array with the indices of the
            # instances.
            start_idx = dataset[self.group]["craters/crater_list_idx"][idx]
            end_idx = dataset[self.group]["craters/crater_list_idx"][idx + 1]
            ellipse_matrices = torch.tensor(
                dataset[self.group]["craters/A_craters"][start_idx:end_idx]
            )
            ellipse_matrices = ellipse_matrices.reshape(-1, 3, 3)

            a, b = ellipse_axes(ellipse_matrices)
            a, b = a / 2, b / 2
            cx, cy = ellipse_center(ellipse_matrices)
            theta = ellipse_angle(ellipse_matrices)
            ellipses = torch.stack([a, b, cx, cy, theta], dim=-1).reshape(-1, 5)

        boxes = bbox_ellipse(ellipses).reshape(-1, 4)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Remove where area < 1e-1
        area_mask = area > 1e-1
        boxes = boxes[area_mask]
        area = area[area_mask]
        ellipses = ellipses[area_mask]

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
            ellipse_params=ellipses,
        )

        return image, target

    def __len__(self) -> int:
        with h5py.File(self.file_path, "r") as f:
            return len(f[self.group]["images"])


class CraterEllipseDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CraterEllipseDataset, managing training/validation/testing dataset splits.
    """

    def __init__(self, file_path: str, batch_size: int, num_workers: int = 4) -> None:
        """
        Initialize the CraterEllipseDataModule.

        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            batch_size (int): Batch size for the data loaders.
        """
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: CraterEllipseDataset | None = None
        self.val_dataset: CraterEllipseDataset | None = None
        self.test_dataset: CraterEllipseDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Sets up datasets for training, validation, and testing.

        Args:
            stage (str): Stage to set up. Can be one of ('fit', 'test', or None).
        """

        if stage in (None, "fit"):  # Set up training and validation datasets
            self.train_dataset = CraterEllipseDataset(self.file_path, group="training")
            self.val_dataset = CraterEllipseDataset(self.file_path, group="validation")
        if stage in (None, "test"):  # Set up test dataset
            self.test_dataset = CraterEllipseDataset(self.file_path, group="test")

        print(
            f"Data loaded. Train dataset: {len(self.train_dataset)} images, Val dataset: {len(self.val_dataset)} images"  # type: ignore
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: Training data loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: Validation data loader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the test dataset.

        Returns:
            DataLoader: Test data loader.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )
