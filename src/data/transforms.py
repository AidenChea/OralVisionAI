from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
)


PATCH_SIZE = (128, 128, 128)


def get_base_transforms() -> Compose:
    return Compose(
        [
            # Add a channel dimension:
            # [X, Y, Z] -> [1, X, Y, Z]
            EnsureChannelFirstd(
                keys=["image", "label"],
                channel_dim="no_channel",
            ),

            # Convert all nonzero label values into a binary mask.
            Lambdad(
                keys="label",
                func=lambda label: (label > 0).astype("float32"),
            ),

            # Normalize CBCT intensities using robust percentiles.
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            # Crop around the expert lesion mask.
            CropForegroundd(
                keys=["image", "label"],
                source_key="label",
                margin=16,
                allow_smaller=True,
            ),

            # Pad or crop to a consistent model input size.
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=PATCH_SIZE,
            ),

            # Convert NumPy arrays to PyTorch tensors.
            EnsureTyped(
                keys=["image", "label"],
                dtype=None,
            ),
        ]
    )
    