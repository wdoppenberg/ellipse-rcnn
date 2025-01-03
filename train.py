import pytorch_lightning as pl
import typer
import random

from pytorch_lightning.callbacks import ModelCheckpoint
from ellipse_rcnn import EllipseRCNN
from ellipse_rcnn.core.model import EllipseRCNNLightning
from ellipse_rcnn.utils.data.fddb import FDDBLightningDataModule
from pytorch_lightning.callbacks import EarlyStopping

app = typer.Typer()
datamodule = FDDBLightningDataModule("data/FDDB", num_workers=4)


@app.command()
def train_model(
    iterations: int = 1,
    lr: float = None,
    weight_decay: float = None,
    lr_min: float = 1e-5,
    lr_max: float = 1e-3,
    weight_decay_min: float = 1e-5,
    weight_decay_max: float = 1e-2,
):
    if iterations > 1:
        print("Warning: Running with multiple iterations.")

    for iteration in range(iterations):
        sampled_lr = random.uniform(lr_min, lr_max)
        sampled_weight_decay = random.uniform(weight_decay_min, weight_decay_max)
        lr = lr if lr is not None else sampled_lr
        weight_decay = (
            weight_decay if weight_decay is not None else sampled_weight_decay
        )

        print(f"Using parameters - Learning rate: {lr}, Weight decay: {weight_decay}")
        print(f"Starting iteration {iteration + 1}/{iterations}")
        model = EllipseRCNN(ellipse_loss_scale=float(1e-1))
        pl_model = EllipseRCNNLightning(model, lr=lr, weight_decay=weight_decay)

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss_total",
            dirpath="checkpoints",
            filename=r"e={epoch:02d}-loss={val/loss_total:.5f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val/loss_total",
            patience=4,
            mode="min",
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            precision="32-true",  # 32-bit needed for numerical stability
            max_epochs=30,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

        trainer.fit(pl_model, datamodule=datamodule)
        print(f"Completed iteration {iteration + 1}/{iterations}")


if __name__ == "__main__":
    typer.run(train_model)
