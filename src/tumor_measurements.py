from pathlib import Path

import nibabel as nib
import numpy as np


LABEL_PATH = Path(
    "data/DOLCHID/cbct_label/AME_10_CBCT_Label.nii.gz"
)


def main() -> None:
    nifti_label = nib.load(LABEL_PATH)
    label = nifti_label.get_fdata()

    tumor_mask = label > 0

    voxel_count = int(np.count_nonzero(tumor_mask))

    voxel_sizes_mm = nifti_label.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(voxel_sizes_mm))
    tumor_volume_mm3 = voxel_count * voxel_volume_mm3
    tumor_volume_cm3 = tumor_volume_mm3 / 1000

    coordinates = np.argwhere(tumor_mask)

    if coordinates.size == 0:
        print("No labeled tumor voxels found.")
        return

    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)

    dimensions_voxels = maximum - minimum + 1
    dimensions_mm = dimensions_voxels * np.array(voxel_sizes_mm)

    center_voxels = coordinates.mean(axis=0)
    center_mm = center_voxels * np.array(voxel_sizes_mm)

    print("=" * 45)
    print("OralVision AI — Tumor Measurements")
    print("=" * 45)
    print(f"Label file: {LABEL_PATH.name}")
    print(f"Volume shape: {label.shape}")
    print(f"Voxel spacing (mm): {voxel_sizes_mm}")
    print(f"Tumor voxels: {voxel_count:,}")
    print(f"Tumor volume: {tumor_volume_mm3:,.2f} mm³")
    print(f"Tumor volume: {tumor_volume_cm3:,.2f} cm³")
    print()
    print(f"Bounding-box size (voxels): {dimensions_voxels}")
    print(f"Bounding-box size (mm): {dimensions_mm}")
    print(f"Approximate center (voxels): {center_voxels}")
    print(f"Approximate center (mm): {center_mm}")


if __name__ == "__main__":
    main()
    