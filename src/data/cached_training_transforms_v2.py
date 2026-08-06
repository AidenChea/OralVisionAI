"""Persistent-cache transforms for foreground-overprediction experiment v2.

The deterministic section is cached in ``data/cache/training_v2``. Random
crop selection and augmentation remain outside the cache and run on every
training access.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    FgBgToIndicesd,
    Lambdad,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRangePercentilesd,
)


PATCH_SIZE: tuple[int, int, int] = (96, 96, 96)
DEFAULT_CACHE_DIR_V2 = "data/cache/training_v2"

# Intensities are normalized to [0, 1] before index generation. Requiring
# values above 0.05 excludes clipped air while retaining low-density tissue.
IMAGE_TISSUE_THRESHOLD = 0.05


def _binarize_label(
    label: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Convert every positive expert-label value to binary foreground."""
    if isinstance(label, torch.Tensor):
        return (label > 0).to(dtype=torch.float32)
    return (label > 0).astype(np.float32)


def get_deterministic_training_transforms_v2() -> Compose:
    """Return deterministic preprocessing persisted once per training case."""
    return Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                image_only=False,
                ensure_channel_first=False,
            ),
            EnsureChannelFirstd(
                keys=["image", "label"],
                channel_dim="no_channel",
            ),
            Lambdad(
                keys="label",
                func=_binarize_label,
            ),
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            FgBgToIndicesd(
                keys="label",
                image_key="image",
                image_threshold=IMAGE_TISSUE_THRESHOLD,
            ),
        ]
    )


def get_random_training_transforms_v2() -> Compose:
    """Return v2 random sampling and spatial augmentation transforms."""
    return Compose(
        [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1,
                neg=3,
                num_samples=4,
                image_key="image",
                image_threshold=IMAGE_TISSUE_THRESHOLD,
                fg_indices_key="label_fg_indices",
                bg_indices_key="label_bg_indices",
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=2,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.5,
                max_k=3,
                spatial_axes=(0, 1),
            ),
            EnsureTyped(
                keys=["image", "label"],
                dtype=torch.float32,
            ),
        ]
    )


def get_cached_training_transforms_v2() -> Compose:
    """Combine cached deterministic transforms with uncached random ones."""
    return Compose(
        [
            *get_deterministic_training_transforms_v2().transforms,
            *get_random_training_transforms_v2().transforms,
        ]
    )
