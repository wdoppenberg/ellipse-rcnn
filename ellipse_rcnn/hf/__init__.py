"""This contains all the Huggingface specific modules.

Make sure you have `huggingface_hub` installed.
"""

try:
    import huggingface_hub
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The module 'huggingface_hub' is required but not installed. "
        "Please install it with 'pip install huggingface_hub'."
    )

from .model import EllipseRCNN
