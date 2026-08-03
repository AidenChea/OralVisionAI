from src.data.dataset import OralVisionDataset
from src.data.transforms import get_base_transforms


def main() -> None:
    dataset = OralVisionDataset(split="train")
    sample = dataset[0]

    print("Before transforms")
    print("Image shape:", sample["image"].shape)
    print("Label shape:", sample["label"].shape)

    transforms = get_base_transforms()
    transformed = transforms(sample)

    image = transformed["image"]
    label = transformed["label"]

    print("\nAfter transforms")
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Image type:", type(image))
    print("Label type:", type(label))
    print("Image minimum:", float(image.min()))
    print("Image maximum:", float(image.max()))
    print("Tumor voxels:", int((label > 0).sum()))

    assert image.shape == (1, 128, 128, 128)
    assert label.shape == (1, 128, 128, 128)

    print("\nPreprocessing test passed.")


if __name__ == "__main__":
    main()
    