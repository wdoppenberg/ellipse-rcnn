from huggingface_hub import PyTorchModelHubMixin

from ellipse_rcnn import EllipseRCNN


class EllipseRCNNHub(
    EllipseRCNN,
    PyTorchModelHubMixin,
    library_name="ellipse-rcnn",  # type: ignore
    tags=["torch", "cv", "object-detection"],  # type: ignore
    repo_url="https://github.com/wdoppenberg/ellipse-rcnn",  # type: ignore
    docs_url="https://github.com/wdoppenberg/ellipse-rcnn",  # type: ignore
): ...
