import datetime as dt
import os
import uuid

import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm as tq

import src.common.constants as const
from src.common.conics import MaskGenerator
from src.common.surrender import SurRenderer


class DataGenerator(MaskGenerator, SurRenderer):
    def __init__(self, *args, **kwargs):
        super(DataGenerator, self).__init__(**kwargs)

    def image_mask_pair(self, **mask_kwargs):
        return self.generate_image(), self.generate_mask(**mask_kwargs)


def generate(size, **kwargs):
    generator = DataGenerator.from_robbins_dataset(
        diamlims=kwargs["diamlims"],
        ellipse_limit=kwargs["ellipse_limit"],
        arc_lims=kwargs["arc_lims"],
        axis_threshold=kwargs["axis_threshold"],
        fov=kwargs["fov"],
        resolution=kwargs["resolution"],
        filled=kwargs["filled"],
        mask_thickness=kwargs["mask_thickness"],
        instancing=kwargs["instancing"]
    )

    date_dataset = np.empty((size, 3), int)
    images_dataset = np.empty((size, 1, *generator.resolution), np.float32)
    if kwargs["instancing"]:
        masks_dataset = np.empty((size, 1, *generator.resolution), np.int16)
    else:
        masks_dataset = np.empty((size, 1, *generator.resolution), np.bool_)
    position_dataset = np.empty((size, 3, 1), np.float64)
    attitude_dataset = np.empty((size, 3, 3), np.float64)
    sol_incidence_dataset = np.empty((size, 1), np.float16)

    A_craters = []

    for i in tq(range(size), desc="Creating dataset"):
        date = dt.date(2021, np.random.randint(1, 12), 1)
        generator.set_random_position()
        generator.scene_time = date
        date_dataset[i] = np.array((date.year, date.month, date.day))

        while not (kwargs["min_sol_incidence"] <= generator.solar_incidence_angle <= kwargs["max_sol_incidence"]):
            generator.set_random_position()  # Generate random position

        position_dataset[i] = generator.position
        sol_incidence_dataset[i] = generator.solar_incidence_angle

        generator.point_nadir()
        if kwargs["randomized_orientation"]:
            # Rotations are incremental (order matters)
            generator.rotate('roll', np.random.randint(0, 360))
            generator.rotate('pitch', np.random.randint(-30, 30))
            generator.rotate('yaw', np.random.randint(-30, 30))

        attitude_dataset[i] = generator.attitude

        image, mask = generator.image_mask_pair()

        masks_dataset[i] = mask[None, None, ...]
        images_dataset[i] = image[None, None, ...]

        if kwargs["save_craters"]:
            A_craters.append(generator.craters_in_image())

    return images_dataset, masks_dataset, position_dataset, attitude_dataset, date_dataset, sol_incidence_dataset, A_craters


def demo_settings(n_demo=20,
                  generation_kwargs=None):
    generation_kwargs_ = const.GENERATION_KWARGS
    if generation_kwargs is not None:
        generation_kwargs_.update(generation_kwargs)
    images, mask, _, _, _, _, _ = generate(n_demo, **generation_kwargs_)

    fig, axes = plt.subplots(n_demo, 2, figsize=(10, 5 * n_demo))
    for i in range(n_demo):
        axes[i, 0].imshow(images[i, 0], cmap='Greys_r')
        axes[i, 1].imshow(mask[i, 0], cmap='gray')
    plt.tight_layout()
    plt.show()


def make_dataset(n_training,
                 n_validation,
                 n_testing,
                 output_path=None,
                 identifier=None,
                 generation_kwargs=None):
    if output_path is None:
        if identifier is not None:
            output_path = f"data/dataset_{identifier}.h5"
        else:
            output_path = "data/dataset_crater_detection.h5"

    generation_kwargs_ = const.GENERATION_KWARGS
    if generation_kwargs is not None:
        generation_kwargs_.update(generation_kwargs)

    if os.path.exists(output_path):
        raise ValueError(f"Dataset named `{os.path.basename(output_path)}` already exists!")

    with h5py.File(output_path, 'w') as hf:
        g_header = hf.create_group("header")

        for k, v in generation_kwargs_.items():
            g_header.create_dataset(k, data=v)

        for group_name, dset_size in zip(
                ("training", "validation", "test"),
                (n_training, n_validation, n_testing)
            ):
            print(f"Creating dataset '{group_name}' @ {dset_size} images")
            group = hf.create_group(group_name)

            (images, masks, position, attitude, date, sol_incidence, A_craters) = generate(dset_size,
                                                                                           **generation_kwargs_)
            for ds, name in zip(
                    (images, masks, position, attitude, date, sol_incidence),
                    ("images", "masks", "position", "attitude", "date", "sol_incidence")
                ):
                group.create_dataset(name, data=ds)

            lengths = np.array([len(cs) for cs in A_craters])
            crater_list_idx = np.insert(lengths.cumsum(), 0, 0)
            A_craters = np.concatenate(A_craters)

            cg = group.create_group("craters")
            cg.create_dataset("crater_list_idx", data=crater_list_idx)
            cg.create_dataset("A_craters", data=A_craters)


def inspect_dataset(dataset_path, plot=True, summary=True, n_inspect=25, pixel_range=(0, 1), return_fig=False):
    with h5py.File(dataset_path, "r") as hf:
        idx = np.random.choice(np.arange(len(hf['training/images'])), n_inspect)
        idx = np.sort(idx)
        images = hf['training/images'][idx]
        masks = hf['training/masks'][idx]
        header = hf["header"]
        header_dict = dict()
        for k, v in header.items():
            header_dict[k] = v[()]

        if summary:
            print("Dataset header:")
            for k, v in header_dict.items():
                print(f"\t{k}: {v}")

    if plot:

        fig, axes = plt.subplots(n_inspect, 3, figsize=(15, 5 * n_inspect))
        n_bins = 256

        for i in range(n_inspect):
            axes[i, 0].imshow(images[i, 0], cmap='gray')

            weights = np.ones_like(images[i, 0].flatten()) / float(len(images[i, 0].flatten()))
            axes[i, 1].hist(images[i, 0].flatten(), n_bins, pixel_range, color='r', weights=weights)
            axes[i, 1].set_xlim(pixel_range)
            axes[i, 1].set_ylabel('Probability')
            axes[i, 1].set_xlabel('Pixel value')

            axes[i, 2].imshow(masks[i][0] * 10, cmap='Blues')

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
    else:
        return header_dict
