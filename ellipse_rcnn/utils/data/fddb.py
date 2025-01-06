"""
Data loader and module for the FDDB dataset.
https://vis-www.cs.umass.edu/fddb/
"""

from glob import glob
from typing import Any
from pathlib import Path

import torch
import pandas as pd
import PIL.Image
import torchvision.transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from ellipse_rcnn.utils.types import TargetDict, ImageTargetTuple, EllipseTuple
from ellipse_rcnn.utils.conics import bbox_ellipse, ellipse_to_conic_matrix, conic_center, unimodular_matrix
from ellipse_rcnn.utils.data.base import EllipseDatasetBase, collate_fn


def preprocess_label_files(root_path: str) -> dict[str, list[EllipseTuple]]:
    label_files = glob(f"{root_path}/labels/*.txt")

    file_paths = []
    ellipse_data = []

    for filename in label_files:
        with open(filename) as f:
            if "ellipseList" not in filename:
                file_paths += [p.strip("\n") for p in f.readlines()]
            else:
                ellipse_data += [p.strip("\n") for p in f.readlines()]

    pdf_file_paths = pd.DataFrame({"path": file_paths})
    pdf_file_paths["path_idx"] = pdf_file_paths.index

    pdf_ellipse_data = pd.DataFrame({"data": ellipse_data})
    pdf_ellipse_data["data_idx"] = pdf_ellipse_data.index

    pdf_file_data_mapping = pdf_file_paths.merge(
        pdf_ellipse_data, left_on="path", right_on="data", how="left"
    )

    ellipse_dict: dict[str, list[EllipseTuple]] = {
        str(k): [] for k in pdf_file_paths["path"]
    }

    for i, r in pdf_file_data_mapping.iterrows():
        data_idx = r["data_idx"]
        num_ellipses = int(ellipse_data[data_idx + 1])
        file_path = r["path"]
        for j in range(data_idx + 2, data_idx + num_ellipses + 2):
            a, b, theta, x, y = [
                float(v) for v in ellipse_data[j].split(" ")[:-1] if len(v) > 0
            ]
            ellipse_params = EllipseTuple(a, b, theta, x, y)
            ellipse_dict[file_path].append(ellipse_params)

    return ellipse_dict


class FDDB(EllipseDatasetBase):
    def __init__(
        self,
        root_path: str | Path,
        ellipse_dict: dict[str, list[EllipseTuple]] | None = None,
        transform: Any = None,
    ) -> None:
        self.root_path = Path(root_path) if isinstance(root_path, str) else root_path
        if transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform
        self.ellipse_dict = ellipse_dict or preprocess_label_files(root_path)

    def __len__(self) -> int:
        return len(self.ellipse_dict)

    def load_target_dict(self, index: int) -> TargetDict:
        key = list(self.ellipse_dict.keys())[index]
        ellipses_list = self.ellipse_dict[key]

        a = torch.tensor([[e.a for e in ellipses_list]])
        b = torch.tensor([[e.b for e in ellipses_list]])
        theta = torch.tensor([[e.theta for e in ellipses_list]])
        x = torch.tensor([[e.x for e in ellipses_list]])
        y = torch.tensor([[e.y for e in ellipses_list]])

        ellipse_matrices = ellipse_to_conic_matrix(a=a, b=b, x=x, y=y, theta=theta)

        if torch.stack(conic_center(ellipse_matrices)).isnan().any():
            raise ValueError("NaN values in ellipse matrices. Please check the data.")

        if len(ellipse_matrices.shape) == 2:
            ellipse_matrices = ellipse_matrices.unsqueeze(0)

        boxes = bbox_ellipse(ellipse_matrices, box_type="xyxy")

        num_objs = len(boxes)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = TargetDict(
            boxes=boxes,
            labels=labels,
            image_id=image_id,
            area=area,
            iscrowd=iscrowd,
            ellipse_matrices=ellipse_matrices,
        )

        return target

    def load_image(self, index: int) -> PIL.Image.Image:
        key = list(self.ellipse_dict.keys())[index]
        file_path = str(Path(self.root_path) / "images" / Path(key)) + ".jpg"
        return PIL.Image.open(file_path)

    def __getitem__(self, idx: int) -> ImageTargetTuple:
        image = self.load_image(idx)
        target_dict = self.load_target_dict(idx)

        # If the image is grayscale, convert it to RGB
        if image.mode == "L":
            image = image.convert("RGB")

        image = self.transform(image)

        return image, target_dict

    def __repr__(self) -> str:
        return f"FDDB<img={len(self)}>"

    def split(self, fraction: float, shuffle: bool = False) -> tuple["FDDB", "FDDB"]:
        """
        Splits the dataset into two subsets based on the given fraction.

        Args:
            fraction (float): Fraction of the dataset for the first subset (0 < fraction < 1).
            shuffle (bool): If True, dataset keys will be shuffled before splitting.

        Returns:
            tuple[FDDB, FDDB]: Two FDDB instances, one with the fraction of data,
                               and the other with the remaining data.
        """
        if not (0 < fraction < 1):
            raise ValueError("The fraction must be between 0 and 1.")

        keys = list(self.ellipse_dict.keys())
        if shuffle:
            import random

            random.shuffle(keys)

        total_length = len(keys)
        split_index = int(total_length * fraction)

        subset1_keys = keys[:split_index]
        subset2_keys = keys[split_index:]

        subset1_ellipse_dict = {key: self.ellipse_dict[key] for key in subset1_keys}
        subset2_ellipse_dict = {key: self.ellipse_dict[key] for key in subset2_keys}

        subset1 = FDDB(
            self.root_path, ellipse_dict=subset1_ellipse_dict, transform=self.transform
        )
        subset2 = FDDB(
            self.root_path, ellipse_dict=subset2_ellipse_dict, transform=self.transform
        )

        return subset1, subset2


class FDDBLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        train_fraction: float = 0.8,
        transform: Any = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.transform = transform
        self.dataset: FDDB | None = None
        self.train_dataset = None
        self.val_dataset = None
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # Ensure data preparation or downloading is done here.
        pass

    def setup(self, stage: str | None = None) -> None:
        # Instantiate the FDDB dataset and split it into training and validation subsets.
        self.dataset = FDDB(self.data_dir, transform=self.transform)

        train_size = int(len(self.dataset) * self.train_fraction)
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader[ImageTargetTuple]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[ImageTargetTuple]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[ImageTargetTuple]:
        # Placeholder for test data; currently returns the validation dataloader as a default.
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )
