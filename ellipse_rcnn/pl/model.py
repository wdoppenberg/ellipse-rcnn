from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor

from ellipse_rcnn import EllipseRCNN
from ellipse_rcnn.core.types import CollatedBatchType


class EllipseRCNNModule(EllipseRCNN, pl.LightningModule):
    def __init__(
        self, lr: float = 1e-4, weight_decay: float = 1e-4, **model_kwargs: Any
    ):
        super().__init__(**model_kwargs)
        self.save_hyperparameters("lr", "weight_decay")
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss_total"},
        }

    def training_step(self, batch: CollatedBatchType, batch_idx: int = 0) -> Tensor:
        images, targets = batch
        loss_dict = self(images, targets)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
        )

        loss = sum(loss_dict.values())
        self.log("train/loss_total", loss, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch: CollatedBatchType, batch_idx: int = 0) -> Tensor:
        self.train(True)
        images, targets = batch

        loss_dict = self(images, targets)

        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        val_loss = sum(loss_dict.values())
        self.log(
            "val/loss_total",
            val_loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "hp_metric",
            val_loss,
        )

        self.log(
            "lr",
            self.lr_schedulers().get_last_lr()[0],
        )

        return val_loss
