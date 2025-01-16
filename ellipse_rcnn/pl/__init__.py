"""This contains all the PyTorch Lightning specific modules.

Make sure you have `pytorch-lightning` installed.
"""

try:
    import pytorch_lightning
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The module 'pytorch_lightning' is required but not installed. "
        "Please install it with 'pip install pytorch_lightning'."
    )

from .model import EllipseRCNNModule
