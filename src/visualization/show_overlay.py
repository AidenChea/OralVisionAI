from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Load image
image_path = Path("data/DOLCHID/cbct_image/AME_10_CBCT_Image.nii.gz")
image = nib.load(image_path).get_fdata()

# Load segmentation mask
label_path = Path("data/DOLCHID/cbct_label/AME_10_CBCT_Label.nii.gz")
label = nib.load(label_path).get_fdata()

# Verify they match
assert image.shape == label.shape, "Image and label dimensions do not match!"

print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")

# Find the slice with the largest amount of tumor
tumor_per_slice = np.sum(label > 0, axis=(0, 1))
best_slice = np.argmax(tumor_per_slice)

print(f"Best slice: {best_slice}")

plt.figure(figsize=(8, 8))

plt.imshow(np.rot90(image[:, :, best_slice]), cmap="gray")

mask = np.rot90(label[:, :, best_slice])

plt.imshow(
    np.ma.masked_where(mask == 0, mask),
    cmap="autumn",
    alpha=0.6,
)

plt.title(f"Tumor Overlay - Slice {best_slice}")
plt.axis("off")
plt.show()
