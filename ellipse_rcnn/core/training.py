import math
import time
from statistics import mean

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm as tq

from utils.data import get_dataloaders
from .metrics import gaussian_angle_distance
from ..utils.conics import conic_center


def train_model(model: nn.Module, num_epochs: int, dataset_path: str, initial_lr=1e-2, run_id: str = None,
                scheduler=None, batch_size: int = 32, momentum: float = 0.9, weight_decay: float = 0.0005,
                num_workers: int = 4, device=None) -> None:
    pretrained = run_id is not None

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(device, str):
        device = torch.device(device)

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
    print(f"MLflow run ID:\n\t{run_id}")

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
            # mlflow.log_metric("train_" + k, mean(v[(e - 1) * len(train_loader):e * len(train_loader)]), step=e)

        for k, v in run_metrics["valid"].items():
            if k == "batch":
                continue
            # mlflow.log_metric("valid_" + k, mean(v[(e - 1) * len(validation_loader):e * len(validation_loader)]),
            #                   step=e)
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

        if len(m2) > 1:
            for pos, d in zip(m2.squeeze().numpy(), dist.numpy()):
                plt.text(*pos, f"{d:.2f}", color='red')

        # mlflow.log_figure(fig, f"sample_output_e{e:02}.png")

        print(
            f"\nSummary:\n",
            f"\tEpoch: {e}/{num_epochs + start_e - 1}\n",
            f"\tAverage train loss: {mean(run_metrics['train']['loss_total'][(e - 1) * len(train_loader):e * len(train_loader)])}\n",
            f"\tAverage validation loss: {mean(run_metrics['valid']['loss_total'][(e - 1) * len(validation_loader):e * len(validation_loader)])}\n",
            f"\tDuration: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f'-----Epoch {e} finished.-----\n')
