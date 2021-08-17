import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as tq

from . import EllipseRCNN
from .metrics import detection_metrics
from .training import CraterEllipseDataset, collate_fn


class Evaluator:
    def __init__(self, model=None, device="cpu", dataset_path="data/dataset_crater_detection.h5"):
        if model is None:
            self._model = EllipseRCNN()
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
