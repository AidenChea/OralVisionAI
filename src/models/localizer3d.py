"""Lightweight 3D U-Net for coarse lesion localization.

Uses a smaller channel configuration than the fine segmentation network so
full 128^3 volumes fit comfortably on an 8 GB GPU during mixed-precision
training.

Research use only — not for clinical diagnosis or treatment.
"""

from __future__ import annotations

from monai.networks.nets import UNet


def create_localizer3d() -> UNet:
    """Create the Experiment V4 coarse localizer network."""
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
        dropout=0.0,
    )
