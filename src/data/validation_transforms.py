"""Deterministic full-volume transforms for validation.

Validation never uses the expert label to select or crop image regions.
The intensity and label preprocessing matches the training pipeline.
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
    ScaleIntensityRangePercentilesd,
)


def _binarize_label(
    label: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Convert all positive expert-label values to binary foreground."""
    if isinstance(label, torch.Tensor):
        return (label > 0).to(dtype=torch.float32)
    return (label > 0).astype(np.float32)


def get_validation_transforms() -> Compose:
    """Return deterministic preprocessing for uncropped validation volumes."""
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
            EnsureTyped(
                keys=["image", "label"],
                dtype=torch.float32,
            ),
        ]
    )
