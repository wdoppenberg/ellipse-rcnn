import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ellipse_rcnn import EllipseRCNN
from ellipse_rcnn.core.model import EllipseRCNNLightning
from ellipse_rcnn.utils.data.base import collate_fn
from ellipse_rcnn.utils.data.fddb import FDDB

if __name__ == "__main__":
    model = EllipseRCNN()
    pl_model = EllipseRCNNLightning(model)

    # Add tensorboard logger
    tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir="tb_logs",
        name="FDDB"
    )

    train_loader = DataLoader(
        FDDB("data/FDDB"),
        batch_size=1,
        num_workers=4,
        collate_fn=collate_fn,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        precision=16,
        auto_lr_find=True,
        max_epochs=10,
        logger=tensorboard_logger,
    )

    trainer.fit(pl_model, train_loader)
