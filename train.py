import pytorch_lightning as pl

from ellipse_rcnn import EllipseRCNN
from ellipse_rcnn.core.model import EllipseRCNNLightning
from ellipse_rcnn.utils.data.fddb import FDDB, FDDBLightningDataModule

if __name__ == "__main__":
    model = EllipseRCNN()
    pl_model = EllipseRCNNLightning(model, lr=1e-5)

    ds = FDDB("data/FDDB")

    datamodule = FDDBLightningDataModule("data/FDDB", num_workers=4)

    trainer = pl.Trainer(
        accelerator="gpu",
        precision="16-mixed",
        max_epochs=15,
        enable_checkpointing=True,
    )

    trainer.fit(pl_model, datamodule=datamodule)
