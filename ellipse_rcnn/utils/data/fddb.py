"""
Data loader and module for the FDDB dataset.
https://vis-www.cs.umass.edu/fddb/
"""
from glob import glob
from typing import Any, Dict, List, NamedTuple
from pathlib import Path

import torch
import pandas as pd
import PIL.Image
import torchvision.transforms

from ellipse_rcnn.utils.types import TargetDict, ImageTargetTuple
from ellipse_rcnn.utils.conics import bbox_ellipse, ellipse_to_conic_matrix
from ellipse_rcnn.utils.data.base import EllipseDatasetBase


class FDDBEllipse(NamedTuple):
    a: float
    b: float
    theta: float
    x: float
    y: float


def preprocess_label_files(root_path: str) -> Dict[str, List[FDDBEllipse]]:
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

    pdf_file_data_mapping = pdf_file_paths.merge(pdf_ellipse_data, left_on="path", right_on="data", how="left")

    ellipse_dict: Dict[str, List[FDDBEllipse]] = {str(k): [] for k in pdf_file_paths["path"]}

    for i, r in pdf_file_data_mapping.iterrows():
        data_idx = r["data_idx"]
        num_ellipses = int(ellipse_data[data_idx + 1])
        file_path = r["path"]
        for j in range(data_idx + 2, data_idx + num_ellipses + 2):
            a, b, theta, x, y = [float(v) for v in ellipse_data[j].split(" ")[:-1] if len(v) > 0]
            ellipse_params = FDDBEllipse(a, b, theta, x, y)
            ellipse_dict[file_path].append(ellipse_params)

    return ellipse_dict


class FDDB(EllipseDatasetBase):
    def __init__(self, root_path: str, transform: Any = None) -> None:
        self.root_path = root_path
        if transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.ellipse_dict = preprocess_label_files(root_path)

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

        ellipse_matrices = ellipse_to_conic_matrix(a, b, theta, x, y)
        if len(ellipse_matrices.shape) == 2:
            ellipse_matrices = ellipse_matrices.unsqueeze(0)

        boxes = bbox_ellipse(ellipse_matrices)

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
        return f"FDDB Dataset with {len(self)} images"
