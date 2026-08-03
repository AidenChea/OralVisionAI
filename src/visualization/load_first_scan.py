from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main() -> None:
    image_folder = Path("data/DOLCHID/cbct_image")

    scan_files = sorted(image_folder.glob("*.nii.gz"))

    if not scan_files:
        raise FileNotFoundError(
            "No .nii.gz scans were found in data/DOLCHID/cbct_image"
        )

    scan_path = scan_files[0]
    print(f"Loading: {scan_path.name}")

    nifti_image = nib.load(scan_path)
    scan = nifti_image.get_fdata()

    print(f"Shape: {scan.shape}")
    print(f"Data type: {scan.dtype}")
    print(f"Minimum value: {np.min(scan):.2f}")
    print(f"Maximum value: {np.max(scan):.2f}")

    middle_slice_index = scan.shape[2] // 2
    middle_slice = scan[:, :, middle_slice_index]

    plt.imshow(np.rot90(middle_slice), cmap="gray")
    plt.title(
        f"{scan_path.name}\nSlice {middle_slice_index} of {scan.shape[2]}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    