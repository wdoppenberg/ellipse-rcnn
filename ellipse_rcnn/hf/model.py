from huggingface_hub import PyTorchModelHubMixin

from ellipse_rcnn import EllipseRCNN as EllipseRCNNBase


class EllipseRCNN(
    EllipseRCNNBase,
    PyTorchModelHubMixin,
    library_name="ellipse-rcnn",
    repo_url="https://github.com/wdoppenberg/ellipse-rcnn",
    docs_url="https://github.com/wdoppenberg/ellipse-rcnn",
): ...
