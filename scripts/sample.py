import typer
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn import EllipseRCNN
from ellipse_rcnn.core.model import EllipseRCNNLightning
from ellipse_rcnn.utils.viz import plot_ellipses, plot_bboxes

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def predict(
    model_path: str = typer.Argument(..., help="Path to the model checkpoint."),
    data_path: str = typer.Argument(..., help="Path to the dataset directory."),
    min_score: float = typer.Option(
        0.6, help="Minimum score threshold for predictions."
    ),
    plot_centers: bool = typer.Option(False, help="Whether to plot ellipse centers."),
) -> None:
    """
    Load a pretrained model, predict ellipses on the given dataset, and visualize results.
    """
    # Load the pretrained model
    typer.echo(f"Loading model from {model_path}...")
    model = EllipseRCNN()
    _ = EllipseRCNNLightning.load_from_checkpoint(model_path, model=model)
    model.eval().cpu()

    # Load the FDDB dataset
    typer.echo(f"Loading dataset from {data_path}...")
    ds = FDDB(data_path)
    ds_raw = FDDB(data_path)
    ds_raw.transform = lambda x: x

    import random

    # Make predictions
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 6, figsize=(16, 4))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        idx = random.randint(0, len(ds))
        image, target = ds[idx]
        image_raw, _ = ds_raw[idx]
        pred = model(image.unsqueeze(0))
        score_mask = pred[0]["scores"] > min_score
        if not len(pred[0]["boxes"][score_mask]) > 0:
            typer.echo(f"No predictions detected for sampled image {idx + 1}.")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.imshow(np.array(image_raw))

        ellipses = pred[0]["ellipse_params"][score_mask].view(-1, 5)
        boxes = pred[0]["boxes"][score_mask].view(-1, 4)

        # Plot ellipses
        plot_ellipses(
            target["ellipse_params"],
            ax=ax,
            plot_centers=plot_centers,
            rim_color="b",
            alpha=1,
        )

        plot_ellipses(ellipses, ax=ax, plot_centers=plot_centers, alpha=0.7)

        # Plot bounding boxes
        plot_bboxes(
            target["boxes"],
            box_type="xyxy",
            ax=ax,
            rim_color="b",
            alpha=1,
        )
        plot_bboxes(boxes, box_type="xyxy", ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
