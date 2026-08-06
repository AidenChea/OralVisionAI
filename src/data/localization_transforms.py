"""Deterministic transforms for coarse full-volume lesion localization.

Research use only. This pipeline intentionally avoids lesion-centered crops and
random augmentation so the localizer sees the entire CBCT volume, resized to a
fixed 128 x 128 x 128 grid.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityRangePercentilesd,
)


LOCALIZATION_SIZE: tuple[int, int, int] = (128, 128, 128)


def _binarize_label(label: np.ndarray) -> np.ndarray:
    """Convert any positive lesion label to a binary float32 mask."""
    return (label > 0).astype(np.float32)


def get_localization_transforms() -> Compose:
    """Return deterministic preprocessing for coarse lesion localization.

    Image volumes use trilinear interpolation. Labels use nearest-neighbor
    interpolation to preserve discrete classes.
    """
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
            Resized(
                keys=["image", "label"],
                spatial_size=LOCALIZATION_SIZE,
                mode=("trilinear", "nearest"),
                align_corners=(False, None),
            ),
            EnsureTyped(
                keys=["image", "label"],
                dtype=torch.float32,
            ),
        ]
    )
