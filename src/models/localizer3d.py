"""Lightweight 3D U-Net used for coarse lesion localization."""

from monai.networks.nets import UNet


def create_localizer3d() -> UNet:
    """Build the coarse 128-cubed lesion-localization network."""
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    )
