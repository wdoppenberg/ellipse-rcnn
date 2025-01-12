from ellipse_rcnn.core.types import UncollatedBatchType, CollatedBatchType


def collate_fn(batch: UncollatedBatchType) -> CollatedBatchType:
    """
    Collate function for the :class:`DataLoader`.

    Parameters
    ----------
    batch:
        A batch of data.
    """
    return tuple(zip(*batch))  # type: ignore
