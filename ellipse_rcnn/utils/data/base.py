from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from ellipse_rcnn.utils.types import (
    TargetDict,
    CollatedBatchType,
    UncollatedBatchType,
)


def collate_fn(batch: UncollatedBatchType) -> CollatedBatchType:
    """
    Collate function for the :class:`DataLoader`.

    Parameters
    ----------
    batch:
        A batch of data.
    """
    return tuple(zip(*batch))  # type: ignore


class EllipseDatasetBase(ABC, Dataset):
    @abstractmethod
    def load_image(self, index: int) -> Any:
        """
        Load the image for the given index.

        Parameters
        ----------
        index:
            The index of the image.

        Returns
        -------
        image:
            The raw image.
        """
        pass

    @abstractmethod
    def load_target_dict(self, index: int) -> TargetDict:
        """
        Load the target dict for the given index.

        Parameters
        ----------
        index:
            The index of the target dict.

        Returns
        -------
        target_dict:
            The target dictionary.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
