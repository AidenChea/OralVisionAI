from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


IMAGE_PATH = Path("data/DOLCHID/cbct_image/AME_10_CBCT_Image.nii.gz")
LABEL_PATH = Path("data/DOLCHID/cbct_label/AME_10_CBCT_Label.nii.gz")


class CBCTViewer:
    def __init__(self, image: np.ndarray, label: np.ndarray) -> None:
        if image.shape != label.shape:
            raise ValueError("Image and label shapes do not match.")

        self.image = image
        self.label = label
        self.slice_index = image.shape[2] // 2
        self.show_mask = True

        self.figure, self.axis = plt.subplots(figsize=(8, 8))

        self.figure.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.draw()

    def draw(self) -> None:
        self.axis.clear()

        image_slice = np.rot90(self.image[:, :, self.slice_index])
        label_slice = np.rot90(self.label[:, :, self.slice_index])

        self.axis.imshow(image_slice, cmap="gray")

        if self.show_mask:
            visible_mask = np.ma.masked_where(label_slice == 0, label_slice)

            self.axis.imshow(
                visible_mask,
                cmap="autumn",
                alpha=0.6,
            )

        tumor_pixels = int(np.count_nonzero(label_slice))

        self.axis.set_title(
            f"Slice {self.slice_index + 1} of {self.image.shape[2]}\n"
            f"Tumor pixels: {tumor_pixels} | "
            f"Mask: {'ON' if self.show_mask else 'OFF'}"
        )

        self.axis.axis("off")
        self.figure.canvas.draw_idle()

    def on_scroll(self, event) -> None:
        print("Scroll detected:", event.button)

        if event.button == "up":
            self.slice_index = min(
                self.slice_index + 1,
                self.image.shape[2] - 1,
            )
        elif event.button == "down":
            self.slice_index = max(
                self.slice_index - 1,
                0,
            )

        self.draw()

    def on_key_press(self, event) -> None:
        print("Key detected:", event.key)

        if event.key == "m":
            self.show_mask = not self.show_mask

        elif event.key in ("up", "right"):
            self.slice_index = min(
                self.slice_index + 1,
                self.image.shape[2] - 1,
            )

        elif event.key in ("down", "left"):
            self.slice_index = max(
                self.slice_index - 1,
                0,
            )

        self.draw()


def main() -> None:
    print(f"Loading image: {IMAGE_PATH.name}")
    print(f"Loading label: {LABEL_PATH.name}")

    image = nib.load(IMAGE_PATH).get_fdata()
    label = nib.load(LABEL_PATH).get_fdata()

    print(f"Volume shape: {image.shape}")
    print("Controls:")
    print("  Mouse wheel: move through slices")
    print("  Arrow keys: move through slices")
    print("  M key: toggle tumor mask")

    viewer = CBCTViewer(image, label)

    # Keep a reference to the viewer while the window remains open.
    plt.show()


if __name__ == "__main__":
    main()
    