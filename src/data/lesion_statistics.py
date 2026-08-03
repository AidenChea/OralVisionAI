from pathlib import Path

import nibabel as nib
import numpy as np


LABEL_DIR = Path("data/DOLCHID/cbct_label")


def main() -> None:
    label_files = sorted(LABEL_DIR.glob("*_CBCT_Label.nii.gz"))

    if not label_files:
        raise FileNotFoundError(f"No labels found in {LABEL_DIR}")

    bounding_boxes = []
    voxel_counts = []
    volumes_cm3 = []

    print(f"Analyzing {len(label_files)} lesion masks...")

    for index, label_path in enumerate(label_files, start=1):
        nifti = nib.load(label_path)
        label = nifti.get_fdata()
        mask = label > 0

        if not np.any(mask):
            print(f"Warning: empty label in {label_path.name}")
            continue

        coordinates = np.argwhere(mask)
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)

        bbox_voxels = maximum - minimum + 1
        bounding_boxes.append(bbox_voxels)

        voxel_count = int(mask.sum())
        voxel_counts.append(voxel_count)

        spacing = np.asarray(nifti.header.get_zooms()[:3], dtype=float)
        voxel_volume_mm3 = float(np.prod(spacing))
        volumes_cm3.append(voxel_count * voxel_volume_mm3 / 1000.0)

        if index % 25 == 0:
            print(f"Processed {index}/{len(label_files)}")

    bounding_boxes = np.asarray(bounding_boxes)
    voxel_counts = np.asarray(voxel_counts)
    volumes_cm3 = np.asarray(volumes_cm3)

    print("\n" + "=" * 55)
    print("OralVision AI — Lesion Statistics")
    print("=" * 55)

    print("\nBounding-box dimensions in voxels [X, Y, Z]")
    print("Minimum:       ", bounding_boxes.min(axis=0))
    print("Median:        ", np.median(bounding_boxes, axis=0).round(1))
    print("90th percentile:", np.percentile(bounding_boxes, 90, axis=0).round(1))
    print("95th percentile:", np.percentile(bounding_boxes, 95, axis=0).round(1))
    print("Maximum:       ", bounding_boxes.max(axis=0))

    print("\nApproximate bounding-box dimensions in millimeters")
    print("Median:        ", (np.median(bounding_boxes, axis=0) * 0.3).round(1))
    print(
        "95th percentile:",
        (np.percentile(bounding_boxes, 95, axis=0) * 0.3).round(1),
    )
    print("Maximum:       ", (bounding_boxes.max(axis=0) * 0.3).round(1))

    print("\nLesion volume")
    print(f"Minimum:        {volumes_cm3.min():.2f} cm³")
    print(f"Median:         {np.median(volumes_cm3):.2f} cm³")
    print(f"95th percentile:{np.percentile(volumes_cm3, 95):.2f} cm³")
    print(f"Maximum:        {volumes_cm3.max():.2f} cm³")

    print(f"\nCases analyzed: {len(bounding_boxes)}")


if __name__ == "__main__":
    main()
    