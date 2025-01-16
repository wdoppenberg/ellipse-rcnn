import random

import pytorch_lightning as pl
import typer
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from ellipse_rcnn.pl import EllipseRCNNModule
from ellipse_rcnn.data.craters import CraterEllipseDataModule
from ellipse_rcnn.data.fddb import FDDBLightningDataModule

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train_model(
    iterations: int = typer.Option(1, help="Number of iterations to train the model."),
    lr: float | None = typer.Option(
        None, help="Learning rate value. Disables lr sampling."
    ),
    weight_decay: float | None = typer.Option(
        None, help="Weight decay value. Disables weight_decay sampling."
    ),
    lr_min: float = typer.Option(1e-5, help="Minimum learning rate for sampling."),
    lr_max: float = typer.Option(1e-3, help="Maximum learning rate for sampling."),
    weight_decay_min: float = typer.Option(
        1e-5, help="Minimum weight decay for sampling."
    ),
    weight_decay_max: float = typer.Option(
        1e-3, help="Maximum weight decay for sampling."
    ),
    num_workers: int = typer.Option(4, help="Number of workers for data loading."),
    batch_size: int = typer.Option(16, help="Batch size for training."),
    dataset: str = typer.Option("FDDB", help="Dataset to use for training."),
    accelerator: str = typer.Option("auto", help="Type of accelerator to use."),
) -> None:
    datamodule: LightningDataModule
    match dataset:
        case "FDDB":
            datamodule = FDDBLightningDataModule(
                "data/FDDB", num_workers=num_workers, batch_size=batch_size
            )

        case "Craters":
            datamodule = CraterEllipseDataModule(
                "data/Craters/dataset_crater_detection_80k.h5",
                batch_size=batch_size,
                num_workers=num_workers,
            )

        case _:
            raise ValueError(f"Dataset {dataset} not found.")

    if iterations > 1 and (lr is not None or weight_decay is not None):
        print(
            "Warning: Running with multiple iterations with a fixed learning rate or weight decay."
        )

    for iteration in range(iterations):
        sampled_lr = random.uniform(lr_min, lr_max)
        sampled_weight_decay = random.uniform(weight_decay_min, weight_decay_max)
        lr = lr if lr is not None else sampled_lr
        weight_decay = (
            weight_decay if weight_decay is not None else sampled_weight_decay
        )

        print(f"Using parameters - Learning rate: {lr}, Weight decay: {weight_decay}")
        print(f"Starting iteration {iteration + 1}/{iterations}")
        pl_module = EllipseRCNNModule(lr=lr, weight_decay=weight_decay)

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss_total",
            dirpath="checkpoints",
            filename=r"loss={val/loss_total:.5f}-e={epoch:02d}",
            auto_insert_metric_name=False,
            save_top_k=1,
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val/loss_total",
            patience=5,
            mode="min",
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            precision="bf16-mixed",
            max_epochs=40,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

        trainer.fit(pl_module, datamodule=datamodule)
        print(f"Completed iteration {iteration + 1}/{iterations}")


if __name__ == "__main__":
    app()
