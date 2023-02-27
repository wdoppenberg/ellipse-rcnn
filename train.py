import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ellipse_rcnn import EllipseRCNN
from ellipse_rcnn.core.model import EllipseRCNNLightning
from ellipse_rcnn.utils.data.base import collate_fn
from ellipse_rcnn.utils.data.fddb import FDDB

if __name__ == "__main__":
    model = EllipseRCNN()
    pl_model = EllipseRCNNLightning(model)

    train_loader = DataLoader(
        FDDB("data/FDDB"),
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn
    )

    trainer = pl.Trainer(accelerator='gpu', precision=16)

    trainer.fit(pl_model, train_loader)
