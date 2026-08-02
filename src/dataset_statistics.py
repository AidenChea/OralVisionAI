from pathlib import Path

import nibabel as nib
import numpy as np

IMAGE_DIR = Path("data/DOLCHID/cbct_image")


def main():
    files = sorted(IMAGE_DIR.glob("*.nii.gz"))

    shapes = []
    spacings = []

    print(f"Scanning {len(files)} CBCT volumes...\n")

    for file in files:
        nii = nib.load(file)

        shapes.append(nii.shape)
        spacings.append(nii.header.get_zooms()[:3])

    shapes = np.array(shapes)
    spacings = np.array(spacings)

    print("=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    print("\nImage Shapes")
    print("Min :", shapes.min(axis=0))
    print("Max :", shapes.max(axis=0))
    print("Mean:", shapes.mean(axis=0).round(1))

    print("\nVoxel Spacing (mm)")
    print("Min :", spacings.min(axis=0))
    print("Max :", spacings.max(axis=0))
    print("Mean:", spacings.mean(axis=0).round(3))


if __name__ == "__main__":
    main()
    