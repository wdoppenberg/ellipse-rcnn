import torch

from ellipse_rcnn.core.types import TargetDict, UncollatedBatchType
from ellipse_rcnn.data.utils import collate_fn


def test_collate_fn() -> None:
    num_instances = 10
    num_images = 5

    test_input: UncollatedBatchType = [
        (
            torch.rand(3, 10, 10),
            TargetDict(
                boxes=torch.rand(num_instances, 4),
                labels=torch.randint(10, (num_instances,)),
                image_id=torch.tensor(0),
                area=torch.rand(num_instances),
                iscrowd=torch.randint(2, (num_instances,)),
                ellipse_matrices=torch.rand(num_instances, 3, 3),
            ),
        )
        for _ in range(num_images)
    ]

    test_output = collate_fn(test_input)

    assert len(test_output) == 2
    assert len(test_output[0]) == num_images
    assert len(test_output[1]) == num_images

    assert test_output[0][0].shape == torch.Size([3, 10, 10])
