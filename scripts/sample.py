import typer
from matplotlib import pyplot as plt
import seaborn as sns
import random

from ellipse_rcnn.data.craters import CraterEllipseDataset
from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.pl import EllipseRCNNModule
from ellipse_rcnn.utils.viz import plot_ellipses, plot_bboxes

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def predict(
    model_path: str = typer.Argument(..., help="Path to the model checkpoint."),
    data_path: str = typer.Argument(..., help="Path to the dataset directory."),
    min_score: float = typer.Option(
        0.6, help="Minimum score threshold for predictions."
    ),
    dataset: str = "FDDB",
    plot_centers: bool = typer.Option(False, help="Whether to plot ellipse centers."),
) -> None:
    """
    Load a pretrained model, predict ellipses on the given dataset, and visualize results.
    """
    match dataset:
        case "FDDB":
            ds = FDDB(
                data_path,
            )

        case "Craters":
            ds = CraterEllipseDataset(data_path, group="test")

        case _:
            raise ValueError(f"Dataset {dataset} not found.")

    # Load the pretrained model
    typer.echo(f"Loading model from {model_path}...")
    model = EllipseRCNNModule.load_from_checkpoint(model_path)
    model.eval().cpu()

    # Make predictions
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(6, 6, figsize=(16, 16))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        idx = random.randint(0, len(ds))
        image, target = ds[idx]
        pred = model(image.unsqueeze(0))
        score_mask = pred[0]["scores"] > min_score
        if not len(pred[0]["boxes"][score_mask]) > 0:
            typer.echo(f"No predictions detected for sampled image {idx + 1}.")
        ax.set_aspect("equal")
        ax.axis("off")
        image = image.permute(1, 2, 0) if image.ndim == 3 else image
        ax.imshow(image, cmap="grey")

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
