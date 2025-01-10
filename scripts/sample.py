import typer
import numpy as np
from matplotlib import pyplot as plt

from ellipse_rcnn.utils.data.fddb import FDDB
from ellipse_rcnn import EllipseRCNN
from ellipse_rcnn.core.model import EllipseRCNNLightning
from ellipse_rcnn.utils.viz import plot_ellipses, plot_bboxes

app = typer.Typer()


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
    ds_raw = FDDB(data_path, transform=lambda x: x)

    # Get the specific image and resolution
    import random

    # Make predictions
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    for ax in axs:
        idx = random.randint(0, len(ds))
        image, target = ds[idx]
        image_raw, _ = ds_raw[idx]
        pred = model(image.unsqueeze(0))
        score_mask = pred[0]["scores"] > min_score
        if not len(pred[0]["boxes"][score_mask]) > 0:
            typer.echo(f"No predictions detected for sampled image {idx + 1}.")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.imshow(np.array(image_raw))
        pred = pred[0]["ellipse_params"].view(-1, 5)

        # Plot ellipses
        plot_ellipses(
            target["ellipse_params"],
            ax=ax,
            plot_centers=plot_centers,
            rim_color="b",
        )
        plot_ellipses(pred[score_mask], ax=ax, plot_centers=plot_centers)

        # Plot bounding boxes
        plot_bboxes(
            target["boxes"],
            box_type="xyxy",
            ax=ax,
            rim_color="b",
            alpha=0.5,
        )
        # plot_bboxes(pred[0]["boxes"][score_mask], box_type="xyxy", ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
