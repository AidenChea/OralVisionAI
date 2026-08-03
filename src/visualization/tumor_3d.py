from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


LABEL_PATH = Path(
    "data/DOLCHID/cbct_label/AME_10_CBCT_Label.nii.gz"
)


def main() -> None:
    nifti_label = nib.load(LABEL_PATH)
    label = nifti_label.get_fdata()

    tumor_mask = label > 0

    if not np.any(tumor_mask):
        raise ValueError("No tumor voxels were found in the label.")

    voxel_spacing = np.array(nifti_label.header.get_zooms()[:3])

    # Crop around the tumor to make reconstruction faster.
    coordinates = np.argwhere(tumor_mask)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0) + 1

    margin = 3
    minimum = np.maximum(minimum - margin, 0)
    maximum = np.minimum(maximum + margin, tumor_mask.shape)

    cropped_mask = tumor_mask[
        minimum[0]:maximum[0],
        minimum[1]:maximum[1],
        minimum[2]:maximum[2],
    ]

    print(f"Original mask shape: {tumor_mask.shape}")
    print(f"Cropped mask shape: {cropped_mask.shape}")
    print(f"Voxel spacing: {voxel_spacing}")

    vertices, faces, _, _ = measure.marching_cubes(
        cropped_mask.astype(np.float32),
        level=0.5,
        spacing=tuple(voxel_spacing),
    )

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(
        vertices[faces],
        alpha=0.75,
    )

    axis.add_collection3d(mesh)

    axis.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    axis.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    axis.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title("3D Reconstruction of Expert Tumor Segmentation")

    # Keep the axes visually proportional.
    axis.set_box_aspect(
        (
            np.ptp(vertices[:, 0]),
            np.ptp(vertices[:, 1]),
            np.ptp(vertices[:, 2]),
        )
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    