import numpy as np
import torch
from astropy.coordinates import cartesian_to_spherical
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as tq

from common import constants as const
from common.conics import plot_conics, conic_center
from detection.metrics import detection_metrics, get_matched_idxs, gaussian_angle_distance
from detection.training import CraterEllipseDataset, collate_fn
from src import CraterDetector


class Evaluator:
    def __init__(self, model=None, device="cpu", dataset_path="data/dataset_crater_detection.h5"):
        if model is None:
            self._model = CraterDetector()
            self._model.load_state_dict(torch.load("blobs/CraterRCNN.pth"))
        else:
            self._model = model

        self._model.eval()

        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError("Type for device is incorrect, must be str or torch.device.")

        self._model.to(self.device)

        self.ds = CraterEllipseDataset(file_path=dataset_path, group="test")

    @torch.no_grad()
    def make_grid(self, n_rows=3, n_cols=4, min_class_score=0.75):
        i = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        loader = DataLoader(self.ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

        for row in range(n_rows):
            for col in range(n_cols):
                images, targets = next(iter(loader))
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                pred = self._model(images)

                scores = pred[0]["scores"]

                A_pred = pred[0]["ellipse_matrices"][scores > min_class_score]

                matched_idxs, matched = get_matched_idxs(pred[0]["boxes"][scores > min_class_score],
                                                         targets[0]["boxes"])

                A_target = targets[0]["ellipse_matrices"]
                if len(matched_idxs) > 0:
                    A_matched = A_target[matched_idxs]
                else:
                    A_matched = torch.zeros((0, 3, 3))
                position = targets[0]["position"]
                r, lat, long = cartesian_to_spherical(*position.cpu().numpy())

                dist = gaussian_angle_distance(A_matched[matched], A_pred[matched])
                m1, m2 = map(lambda arr: torch.vstack(tuple(conic_center(arr).T)).T[..., None],
                             (A_matched[matched], A_pred[matched]))

                lat = np.degrees(lat.value)[0]
                long = np.degrees(long.value)[0]
                long -= 360 if long > 180 else 0

                textstr = '\n'.join((
                    rf'$\varphi={lat:.1f}^o$',
                    rf'$\lambda={long:.1f}^o$',
                    rf'$h={r.value[0] - const.RMOON:.0f}$ km',
                ))

                axes[row, col].imshow(images[0][0].cpu().numpy(), cmap='gray')
                axes[row, col].axis("off")
                axes[row, col].set_title(i)

                plot_conics(A_target.cpu(), ax=axes[row, col], rim_color='cyan')
                plot_conics(A_pred.cpu()[matched], ax=axes[row, col], rim_color='red')
                plot_conics(A_pred.cpu()[~matched], ax=axes[row, col], rim_color='yellow')

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                axes[row, col].text(0.05, 0.95, textstr, transform=axes[row, col].transAxes, fontsize=14,
                                    verticalalignment='top', bbox=props)

                if len(m2) > 2:
                    for pos, d in zip(m2.squeeze().cpu().numpy(), dist.cpu().numpy()):
                        axes[row, col].text(*pos, f"{d:.2f}", color='white')

                i += 1
        fig.tight_layout()

        return fig

    @torch.no_grad()
    def performance_metrics(self, iou_threshold=0.5, confidence_thresholds=None, distance_threshold=None):

        if confidence_thresholds is None:
            confidence_thresholds = torch.arange(start=0.05, end=0.99, step=0.05).to(self.device)

        loader = DataLoader(self.ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)

        bar = tq(loader, desc=f"Testing",
                 postfix={
                     "IoU": 0.,
                     "GA_distance": 0.,
                     "precision": 0.,
                     "recall": 0.,
                     "f1_score": 0.
                 })

        precision = torch.zeros((len(loader), loader.batch_size, len(confidence_thresholds)), device=self.device)
        recall = torch.zeros((len(loader), loader.batch_size, len(confidence_thresholds)), device=self.device)
        f1 = torch.zeros((len(loader), loader.batch_size, len(confidence_thresholds)), device=self.device)
        iou = torch.zeros((len(loader), loader.batch_size, len(confidence_thresholds)), device=self.device)
        dist = torch.zeros((len(loader), loader.batch_size, len(confidence_thresholds)), device=self.device)

        for batch, (images, targets_all) in enumerate(bar):
            images = list(image.to(self.device) for image in images)
            targets_all = [{k: v.to(self.device) for k, v in t.items()} for t in targets_all]

            pred_all = self._model(images)

            for i, (pred, targets) in enumerate(zip(pred_all, targets_all)):
                for j, confidence_threshold in enumerate(confidence_thresholds):
                    precision[batch, i, j], recall[batch, i, j], f1[batch, i, j], \
                    iou[batch, i, j], dist[batch, i, j] = detection_metrics(pred,
                                                                            targets,
                                                                            iou_threshold=iou_threshold,
                                                                            confidence_threshold=confidence_threshold,
                                                                            distance_threshold=distance_threshold)

            postfix = dict(
                IoU=iou[batch].mean().item(),
                GA_distance=dist[batch].mean().item(),
                precision=precision[batch].mean().item(),
                recall=recall[batch].mean().item(),
                f1_score=f1[batch].mean().item()
            )
            bar.set_postfix(ordered_dict=postfix)

        del images, targets_all

        precision_out = torch.zeros(len(confidence_thresholds))
        recall_out = torch.zeros(len(confidence_thresholds))
        f1_out = torch.zeros(len(confidence_thresholds))
        iou_out = torch.zeros(len(confidence_thresholds))
        dist_out = torch.zeros(len(confidence_thresholds))

        for i in range(len(confidence_thresholds)):
            precision_out[i], recall_out[i], f1_out[i], iou_out[i], dist_out[i] = map(
                lambda x: x[..., i][x[..., i] > 0.].mean(),
                (precision, recall, f1, iou, dist)
            )

        precision, recall, f1, iou, dist = map(lambda x: x.mean((0, 1)), (precision, recall, f1, iou, dist))
        return precision, recall, f1, iou, dist, confidence_thresholds

    def precision_recall_plot(self, iou_thresholds=None, confidence_thresholds=None):
        fig, ax = plt.subplots(figsize=(10, 5))

        if iou_thresholds is None:
            iou_thresholds = torch.arange(start=0.5, end=0.99, step=0.10)

        for iou_threshold in iou_thresholds:
            precision, recall, f1, iou, dist, confidence_thresholds = self.performance_metrics(
                iou_threshold=iou_threshold)

            R = torch.cat((torch.ones(1), recall.cpu(), torch.zeros(1)))
            P = torch.cat((torch.zeros(1), precision.cpu(), torch.ones(1)))

            AUC = torch.trapz(R, P)

            ax.fill_between(R, P, alpha=0.2, step='pre')
            ax.step(R, P, label=f'IoU>{iou_threshold:.2f} | AUC={AUC:.3f}')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim((0, 1.05))
        ax.set_ylim((0, 1.05))
        ax.legend()

        return fig


if __name__ == "__main__":
    ev = Evaluator(device="cuda")

    # fig = ev.make_grid()
    # plt.show()
    # fig.savefig("output/detection_mosaic.png")
    ev.performance_metrics()
