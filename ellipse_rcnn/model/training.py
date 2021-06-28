import math
import os
import time
from statistics import mean
from typing import Tuple, Dict, Iterable

import h5py
import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm as tq

from src.detection.metrics import gaussian_angle_distance
from src.common.conics import conic_center, plot_conics
from src.common.data import inspect_dataset


class CraterDataset(Dataset):
    def __init__(self,
                 file_path,
                 group
                 ):
        self.file_path = file_path
        self.group = group

    def __getitem__(self, idx: ...) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.file_path, 'r') as dataset:

            image = dataset[self.group]["images"][idx]
            masks = dataset[self.group]["masks"][idx]

            image = torch.as_tensor(image)
            masks = torch.as_tensor(masks, dtype=torch.float32)

            return image, masks

    def random(self):
        return self.__getitem__(
            np.random.randint(0, len(self))
        )

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f[self.group]['images'])


def collate_fn(batch: Iterable):
    return tuple(zip(*batch))


class CraterMaskDataset(CraterDataset):
    def __init__(self, min_area=4, box_padding: float = 0., **kwargs):
        super(CraterMaskDataset, self).__init__(**kwargs)
        self.min_area = min_area
        self.box_padding = box_padding

    def __getitem__(self, idx: ...) -> Tuple[torch.Tensor, Dict]:
        image, mask = super(CraterMaskDataset, self).__getitem__(idx)

        with h5py.File(self.file_path, 'r') as dataset:
            position = torch.as_tensor(dataset[self.group]["position"][idx], dtype=torch.float64)
            attitude = torch.as_tensor(dataset[self.group]["attitude"][idx], dtype=torch.float64)

        mask: torch.Tensor = mask.int()

        obj_ids = mask.unique()[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)

        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)

        for i in range(num_objs):
            pos = torch.where(masks[i])
            xmin = pos[1].min()
            xmax = pos[1].max()
            ymin = pos[0].min()
            ymax = pos[0].max()

            if self.box_padding > 0.:
                dx = xmax - xmin
                dy = ymax - ymin

                xmin -= (dx * self.box_padding).to(xmin)
                xmax += (dx * self.box_padding).to(xmax)
                ymin -= (dy * self.box_padding).to(ymin)
                ymax += (dy * self.box_padding).to(ymax)

            boxes[i] = torch.tensor([xmin, ymin, xmax, ymax])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area_filter = area > self.min_area

        masks, obj_ids, boxes, area = map(lambda x: x[area_filter], (masks, obj_ids, boxes, area))

        num_objs = len(obj_ids)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = masks.int()
        image_id = torch.tensor([idx])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict(
            boxes=boxes,
            labels=labels,
            masks=masks,
            image_id=image_id,
            area=area,
            iscrowd=iscrowd,
            position=position,
            attitude=attitude
        )

        return image, target

    @staticmethod
    def collate_fn(batch: Iterable):
        return collate_fn(batch)


class CraterEllipseDataset(CraterMaskDataset):
    def __init__(self, **kwargs):
        super(CraterEllipseDataset, self).__init__(min_area=0, box_padding=0, **kwargs)

    def __getitem__(self, idx: ...) -> Tuple[torch.Tensor, Dict]:
        image, target = super(CraterEllipseDataset, self).__getitem__(idx)
        target.pop("masks")

        with h5py.File(self.file_path, 'r') as dataset:
            start_idx = dataset[self.group]["craters/crater_list_idx"][idx]
            end_idx = dataset[self.group]["craters/crater_list_idx"][idx + 1]
            A_craters = dataset[self.group]["craters/A_craters"][start_idx:end_idx]

        boxes = target["boxes"]

        x_box = boxes[:, 0] + ((boxes[:, 2] - boxes[:, 0]) / 2)
        y_box = boxes[:, 1] + ((boxes[:, 3] - boxes[:, 1]) / 2)

        x, y = conic_center(A_craters).T

        if len(x_box) > 0 and len(x) > 0:
            matched_idxs = cdist(np.vstack((x_box.numpy(), y_box.numpy())).T, np.vstack((x, y)).T).argmin(1)
            A_craters = A_craters[matched_idxs]
        else:
            A_craters = torch.zeros((0, 3, 3))

        target['ellipse_matrices'] = torch.as_tensor(A_craters, dtype=torch.float32)

        return image, target


def get_dataloaders(dataset_path: str, batch_size: int = 10, num_workers: int = 2) -> \
        Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = CraterEllipseDataset(file_path=dataset_path, group="training")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
                              shuffle=True)

    validation_dataset = CraterEllipseDataset(file_path=dataset_path, group="validation")
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers,
                                   collate_fn=collate_fn, shuffle=True)

    test_dataset = CraterEllipseDataset(file_path=dataset_path, group="test")
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn,
                             shuffle=True)

    return train_loader, validation_loader, test_loader


def train_model(model: nn.Module, num_epochs: int, dataset_path: str, initial_lr=1e-2, run_id: str = None,
                scheduler=None, batch_size: int = 32, momentum: float = 0.9, weight_decay: float = 0.0005,
                num_workers: int = 4, device=None) -> None:

    pretrained = run_id is not None

    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("crater-model")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained:
        checkpoint = mlflow.pytorch.load_state_dict(f"runs:/{run_id}/checkpoint")

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = SGD(params, lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer, patience=5, cooldown=2)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        checkpoint = dict()
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = SGD(params, lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer, patience=5, cooldown=2)

    tracked_params = ('momentum', 'weight_decay', 'dampening')

    name = "Ellipse R-CNN"

    run_args = dict(run_name=name)
    if pretrained:
        start_e = checkpoint['epoch'] + 1
        run_metrics = checkpoint['run_metrics']
        run_args['run_id'] = run_id
    else:
        start_e = 1
        run_metrics = dict(
            train=dict(
                batch=list(),
                loss_total=list(),
                loss_classifier=list(),
                loss_box_reg=list(),
                loss_ellipse=list(),
                loss_objectness=list(),
                loss_rpn_box_reg=list()
            ),
            valid=dict(
                batch=list(),
                loss_total=list(),
                loss_classifier=list(),
                loss_box_reg=list(),
                loss_ellipse=list(),
                loss_objectness=list(),
                loss_rpn_box_reg=list()
            )
        )

    with mlflow.start_run(**run_args) as run:
        run_id = run.info.run_id
        print(f"MLflow run ID:\n\t{run_id}")

        if not pretrained:
            mlflow.log_param('optimizer', type(optimizer).__name__)
            mlflow.log_param('dataset', os.path.basename(dataset_path))
            for tp in tracked_params:
                try:
                    mlflow.log_param(tp, optimizer.state_dict()['param_groups'][0][tp])
                except KeyError as err:
                    pass
            mlflow.log_figure(inspect_dataset(dataset_path, return_fig=True, summary=False), f"dataset_inspection.png")

        for e in range(start_e, num_epochs + start_e):
            train_loader, validation_loader, test_loader = get_dataloaders(dataset_path, batch_size, num_workers)

            print(f'\n-----Epoch {e} started-----\n')

            since = time.time()

            mlflow.log_metric('lr', optimizer.state_dict()['param_groups'][0]['lr'], step=e)

            model.train()
            bar = tq(train_loader, desc=f"Training [{e}]",
                     postfix={
                         "loss_total": 0.,
                         "loss_classifier": 0.,
                         "loss_box_reg": 0.,
                         "loss_ellipse": 0.,
                         "loss_objectness": 0.,
                         "loss_rpn_box_reg": 0
                     })
            for batch, (images, targets) in enumerate(bar, 1):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                loss = sum(l for l in loss_dict.values())

                if not math.isfinite(loss):
                    del images, targets
                    raise RuntimeError(f"Loss is {loss}, stopping training")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                postfix = dict(loss_total=loss.item())
                run_metrics["train"]["loss_total"].append(loss.item())
                run_metrics["train"]["batch"].append(batch)

                for k, v in loss_dict.items():
                    postfix[k] = v.item()
                    run_metrics["train"][k].append(v.item())

                bar.set_postfix(ordered_dict=postfix)

            with torch.no_grad():
                bar = tq(validation_loader, desc=f"Validation [{e}]",
                         postfix={
                             "loss_total": 0.,
                             "loss_classifier": 0.,
                             "loss_box_reg": 0.,
                             "loss_ellipse": 0.,
                             "loss_objectness": 0.,
                             "loss_rpn_box_reg": 0
                         })
                for batch, (images, targets) in enumerate(bar, 1):
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)

                    loss = sum(l for l in loss_dict.values())

                    if not math.isfinite(loss):
                        del images, targets
                        raise RuntimeError(f"Loss is {loss}, stopping validation")

                    postfix = dict(loss_total=loss.item())
                    run_metrics["valid"]["loss_total"].append(loss.item())
                    run_metrics["valid"]["batch"].append(batch)

                    for k, v in loss_dict.items():
                        postfix[k] = v.item()
                        run_metrics["valid"][k].append(v.item())

                    bar.set_postfix(ordered_dict=postfix)

            time_elapsed = time.time() - since

            for k, v in run_metrics["train"].items():
                if k == "batch":
                    continue
                mlflow.log_metric("train_" + k, mean(v[(e - 1) * len(train_loader):e * len(train_loader)]), step=e)

            for k, v in run_metrics["valid"].items():
                if k == "batch":
                    continue
                mlflow.log_metric("valid_" + k, mean(v[(e - 1) * len(validation_loader):e * len(validation_loader)]),
                                  step=e)
            scheduler.step(
                mean(run_metrics["valid"]["loss_total"][(e - 1) * len(validation_loader):e * len(validation_loader)]))

            state_dict = {
                'epoch': e,
                'run_id': run_id,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'run_metrics': run_metrics
            }

            mlflow.pytorch.log_state_dict(state_dict, artifact_path="checkpoint")

            images, targets = next(iter(test_loader))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            model.eval()

            A_craters_pred = model.get_conics(images, min_score=0.75).cpu()
            A_craters_target = targets[0]["ellipse_matrices"].cpu()

            m_target = conic_center(targets[0]["ellipse_matrices"].cpu())
            m_pred = conic_center(A_craters_pred)

            matched_idxs = torch.cdist(m_target, m_pred).argmin(0)

            A_craters_target = A_craters_target[matched_idxs]

            dist = gaussian_angle_distance(A_craters_target, A_craters_pred)
            m1, m2 = map(lambda arr: torch.vstack(tuple(conic_center(arr).T)).T[..., None],
                         (A_craters_pred, A_craters_target))

            fig, ax = plt.subplots(figsize=(10, 10))

            ax.imshow(images[0][0].cpu().numpy(), cmap='gray')
            plot_conics(A_craters_target, ax=ax, rim_color='cyan')
            plot_conics(A_craters_pred, ax=ax)

            if len(m2) > 1:
                for pos, d in zip(m2.squeeze().numpy(), dist.numpy()):
                    plt.text(*pos, f"{d:.2f}", color='red')

            mlflow.log_figure(fig, f"sample_output_e{e:02}.png")

            print(
                f"\nSummary:\n",
                f"\tEpoch: {e}/{num_epochs + start_e - 1}\n",
                f"\tAverage train loss: {mean(run_metrics['train']['loss_total'][(e - 1) * len(train_loader):e * len(train_loader)])}\n",
                f"\tAverage validation loss: {mean(run_metrics['valid']['loss_total'][(e - 1) * len(validation_loader):e * len(validation_loader)])}\n",
                f"\tDuration: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
            print(f'-----Epoch {e} finished.-----\n')
