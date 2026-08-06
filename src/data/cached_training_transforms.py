"""Training transforms split for MONAI ``PersistentDataset`` caching.

Deterministic transforms (cached to disk once per case):
  - Load NIfTI volumes
  - Channel ordering, label binarization, intensity scaling
  - Foreground/background index precomputation for fast cropping

Random transforms (applied on every access):
  - Pos/neg patch sampling, flips, rotations, dtype conversion

``PersistentDataset`` automatically caches everything *before* the first
random transform in the composed pipeline, so random augmentations are
never written to the cache directory.
"""

from __future__ import annotations

import numpy as np
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

# Shared with the original training pipeline.
PATCH_SIZE: tuple[int, int, int] = (96, 96, 96)

DEFAULT_CACHE_DIR = "data/cache/training"


def _binarize_label(label: np.ndarray) -> np.ndarray:
    """Convert lesion labels to binary ``float32`` masks."""
    return (label > 0).astype(np.float32)


def get_deterministic_training_transforms() -> Compose:
    """Expensive, reproducible preprocessing cached by ``PersistentDataset``."""
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
                image_threshold=0.0,
            ),
        ]
    )


def get_random_training_transforms() -> Compose:
    """Stochastic augmentations applied after loading from cache."""
    return Compose(
        [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0.0,
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
            ),
        ]
    )


def get_cached_training_transforms() -> Compose:
    """Full pipeline: deterministic section first, then random section.

    ``PersistentDataset`` splits the compose at the first random transform
    (``RandCropByPosNegLabeld``), so only the deterministic section is
    persisted under ``data/cache/training``.
    """
    return Compose(
        [
            *get_deterministic_training_transforms().transforms,
            *get_random_training_transforms().transforms,
        ]
    )
