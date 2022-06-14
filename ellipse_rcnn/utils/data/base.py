from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from ellipse_rcnn.utils.types import TargetDict, ImageTargetTuple, CollatedBatchType, UncollatedBatchType


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
    def __init__(
        self,
        data_file: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.data_file = data_file

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

    def _transform_all(self, image: Any, target_dict: TargetDict) -> ImageTargetTuple:
        """
        Transform the image and target dict.

        Parameters
        ----------
        image:
            The image.
        target_dict:
            The target dict.

        Returns
        -------
        image:
            The transformed image.
        target_dict:
            The transformed target dict.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> ImageTargetTuple:
        return self._transform_all(self.load_image(index), self.load_target_dict(index))

    @abstractmethod
    def __len__(self) -> int:
        pass
