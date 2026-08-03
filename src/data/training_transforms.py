from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRangePercentilesd,
)


PATCH_SIZE = (96, 96, 96)


def get_training_transforms() -> Compose:
    return Compose(
        [
            EnsureChannelFirstd(
                keys=["image", "label"],
                channel_dim="no_channel",
            ),

            Lambdad(
                keys="label",
                func=lambda label: (label > 0).astype("float32"),
            ),

            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0.0,
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
    