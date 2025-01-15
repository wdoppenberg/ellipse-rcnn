"""
Data loader and module for the FDDB dataset.
https://vis-www.cs.umass.edu/fddb/
"""

import os
from pathlib import Path
from typing import Self

import PIL.Image
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url, extract_archive

from ellipse_rcnn.core.ops import (
    bbox_ellipse,
)
from ellipse_rcnn.core.types import TargetDict, ImageTargetTuple, EllipseTuple
from ellipse_rcnn.data.base import EllipseDatasetBase
from ellipse_rcnn.data.utils import collate_fn


def preprocess_label_files(root_path: Path) -> dict[str, list[EllipseTuple]]:
    label_files = root_path.glob("labels/*.txt")

    file_paths = []
    ellipse_data = []

    for filename in label_files:
        with open(filename) as f:
            if "ellipseList" not in filename.name:
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
            ellipse_params = EllipseTuple(a=a, b=b, x=x, y=y, theta=theta)
            ellipse_dict[file_path].append(ellipse_params)

    return ellipse_dict


class FDDB(EllipseDatasetBase):
    resources = {
        "labels": (
            "http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz",
            None,
        ),
        "images": (
            "http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz",
            None,
        ),
    }

    def __init__(
        self,
        root: str | Path = Path("./data/FDDB"),
        train: bool = True,
        download: bool = False,
        verbose: bool = True,
        ellipse_dict: dict[str, list[EllipseTuple]] | None = None,
    ) -> None:
        """
        Initializes the FDDB dataset object.

        Parameters
        ----------
        root : str or Path, optional
            Root directory of the dataset where ``FDDB/processed/training.pt`` and
            ``FDDB/processed/test.pt`` exist. Defaults to './data/FDDB'.
        train : bool, optional
            If True, creates the dataset from ``training.pt``; otherwise, from ``test.pt``.
            Defaults to True.
        download : bool, optional
            If True, downloads the dataset from the internet and stores it in the root directory.
            Defaults to True.
        verbose : bool, optional
            If True, enables verbose logging. Defaults to False.
        ellipse_dict : dict[str, list[EllipseTuple]], optional
            Dictionary containing the ellipse data for each image. If not provided, it will be
            generated from the label files. Defaults to None.

        Raises
        ------
        RuntimeError
            If the dataset is not found and `download` is False.
        """
        super().__init__()
        self.root: Path = Path(root)
        self.train = train
        self.verbose = verbose

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it")

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        self.ellipse_dict = ellipse_dict or preprocess_label_files(self.root)

    def _check_exists(self) -> bool:
        """
        Check if the dataset has been downloaded and extracted properly.

        Returns
        -------
        bool
            True if the dataset is present, False otherwise.
        """
        images_path = self.root / "images"
        annotations_path = self.root / "labels"
        return images_path.exists() and annotations_path.exists()

    def download(self) -> None:
        """
        Download and extract the FDDB dataset.

        If the dataset already exists, no action is taken.

        Raises
        ------
        OSError
            If there is an issue during the download or extraction.
        """
        if self._check_exists():
            if self.verbose:
                print(f"FDDB Dataset already present under {self.root}.")
            return

        self.root.mkdir(parents=True, exist_ok=True, mode=0o755)

        # Download and extract files

        for subfolder, (url, md5) in self.resources.items():
            filename = os.path.basename(url)
            download_url(url, self.root, filename, md5)
            extract_archive(
                self.root / filename, self.root / subfolder, remove_finished=True
            )

        if self.verbose:
            print("Dataset downloaded and extracted successfully")

    def __len__(self) -> int:
        return len(self.ellipse_dict)

    def load_target_dict(self, index: int) -> TargetDict:
        key = list(self.ellipse_dict.keys())[index]
        ellipses_list = self.ellipse_dict[key]

        a = torch.tensor([[e.a for e in ellipses_list]])
        b = torch.tensor([[e.b for e in ellipses_list]])
        cx = torch.tensor([[e.x for e in ellipses_list]])
        cy = torch.tensor([[e.y for e in ellipses_list]])
        theta = torch.tensor([[e.theta for e in ellipses_list]])

        ellipse_params = torch.stack((a, b, cx, cy, theta), dim=-1).reshape(-1, 5)

        boxes = bbox_ellipse(ellipse_params)

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
            ellipse_params=ellipse_params,
        )

        return target

    def load_image(self, index: int) -> PIL.Image.Image:
        key = list(self.ellipse_dict.keys())[index]
        file_path = str(Path(self.root) / "images" / Path(key)) + ".jpg"
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

    def split(self, fraction: float, shuffle: bool = False) -> tuple[Self, Self]:
        """
        Splits the dataset into two subsets based on the given fraction.

        Parameters
        ----------
        fraction : float
            Fraction of the dataset for the first subset (0 < fraction < 1).
        shuffle : bool, optional
            If True, dataset keys will be shuffled before splitting. Defaults to False.

        Returns
        -------
        tuple of FDDB
            Two FDDB instances, the first having the fraction of data, and
            the other with the remaining data.

        Raises
        ------
        ValueError
            If the fraction is not between 0 and 1.
        """
        if not (0 < fraction < 1):
            raise ValueError("The fraction must be between 0 and 1.")

        keys = list(self.ellipse_dict.keys())
        if shuffle:
            import random

            random.shuffle(keys)

        total_length = len(self)
        split_index = int(total_length * fraction)

        subset1_keys = keys[split_index:]
        subset2_keys = keys[:split_index]

        subset1_ellipse_dict = {key: self.ellipse_dict[key] for key in subset1_keys}
        subset2_ellipse_dict = {key: self.ellipse_dict[key] for key in subset2_keys}

        subset1 = FDDB(self.root, ellipse_dict=subset1_ellipse_dict)
        subset2 = FDDB(
            self.root,
            ellipse_dict=subset2_ellipse_dict,
        )

        return subset1, subset2


class FDDBLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path = Path("./data/FDDB"),
        batch_size: int = 16,
        train_fraction: float = 0.9,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.dataset: FDDB | None = None
        self.train_dataset: FDDB | None = None
        self.val_dataset: FDDB | None = None
        self.test_dataset: FDDB | None = None
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # Ensure data preparation or downloading is done here.
        self.dataset = FDDB(self.data_dir, download=True)

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            self.dataset = FDDB(self.data_dir, download=False)

        # Instantiate the FDDB dataset and split it into training and validation subsets.
        self.train_dataset, self.val_dataset = self.dataset.split(
            1 - self.train_fraction
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None or stage == "validate":
            self.train_dataset = self.train_dataset
            self.val_dataset = self.val_dataset
        elif stage == "test":
            self.test_dataset = self.val_dataset
        else:
            raise ValueError(f"Invalid stage {stage}.")

        print(
            f"Data loaded. Train dataset: {len(self.train_dataset)} images, Val dataset: {len(self.val_dataset)} images"
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
