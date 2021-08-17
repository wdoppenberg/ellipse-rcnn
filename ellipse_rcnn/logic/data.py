from collections import Iterable
from typing import Tuple, Dict

import h5py
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.conics import bbox_ellipse


def collate_fn(batch: Iterable):
    return tuple(zip(*batch))


class EllipseDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 group: str
                 ):
        self.file_path = file_path
        self.group = group

    def __getitem__(self, idx: ...) -> Tuple[torch.Tensor, Dict]:
        with h5py.File(self.file_path, 'r') as dataset:
            image = torch.Tensor(dataset[self.group]["images"][idx])
            start_idx = dataset[self.group]["craters/crater_list_idx"][idx]
            end_idx = dataset[self.group]["craters/crater_list_idx"][idx + 1]
            A_craters = torch.Tensor(dataset[self.group]["craters/A_craters"][start_idx:end_idx])

        boxes = bbox_ellipse(A_craters)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        num_objs = len(boxes)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict(
            boxes=boxes,
            labels=labels,
            image_id=image_id,
            area=area,
            iscrowd=iscrowd,
            ellipse_matrices=A_craters
        )

        return image, target

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f[self.group]['images'])

    @staticmethod
    def collate_fn(batch: Iterable):
        return collate_fn(batch)


def get_dataloaders(dataset_path: str, batch_size: int = 10, num_workers: int = 2) -> \
        Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = EllipseDataset(file_path=dataset_path, group="training")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
                              shuffle=True)

    validation_dataset = EllipseDataset(file_path=dataset_path, group="validation")
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers,
                                   collate_fn=collate_fn, shuffle=True)

    test_dataset = EllipseDataset(file_path=dataset_path, group="test")
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn,
                             shuffle=True)

    return train_loader, validation_loader, test_loader
