import torch

from ellipse_rcnn.utils.data import collate_fn
from ellipse_rcnn.utils.types import TargetDict, UncollatedBatchType


def test_collate_fn() -> None:
    NUM_INSTANCES = 10
    NUM_IMAGES = 5

    test_input: UncollatedBatchType = [
        (
            torch.rand(3, 10, 10),
            TargetDict(
                boxes=torch.rand(NUM_INSTANCES, 4),
                labels=torch.randint(10, (NUM_INSTANCES,)),
                image_id=torch.tensor(0),
                area=torch.rand(NUM_INSTANCES),
                iscrowd=torch.randint(2, (NUM_INSTANCES,)),
                ellipse_matrices=torch.rand(NUM_INSTANCES, 3, 3),
            ),
        )
        for _ in range(NUM_IMAGES)
    ]

    test_output = collate_fn(test_input)

    assert len(test_output) == 2
    assert len(test_output[0]) == NUM_IMAGES
    assert len(test_output[1]) == NUM_IMAGES
