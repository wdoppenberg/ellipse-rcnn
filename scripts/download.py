from pathlib import Path

import typer
from typer import Typer

from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.data.base import EllipseDatasetBase

app = Typer()


AVAILABLE_DATASETS: dict[str, type[EllipseDatasetBase]] = {"FDDB": FDDB}


@app.command()
def download(dataset: str, root: Path = Path("./data/")) -> None:
    """
    Download a dataset.
    """
    try:
        AVAILABLE_DATASETS[dataset](root=root / dataset, download=True)  # type: ignore

    except KeyError:
        raise typer.echo(
            f"Dataset {dataset} not available. Available datasets: {list(AVAILABLE_DATASETS.keys())}"
        )


if __name__ == "__main__":
    app()
