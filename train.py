import pytorch_lightning as pl

from ellipse_rcnn import get_dataloaders, EllipseRCNN

if __name__ == "__main__":
    train_loader, validation_loader, test_loader = get_dataloaders(
        "C:\dev\workspace\crater-detection\data\dataset_crater_detection_80k.h5",
        batch_size=32,
        num_workers=8
    )

    model = EllipseRCNN()

    trainer = pl.Trainer(gpus=1, precision=16)

    trainer.fit(model, train_loader)
