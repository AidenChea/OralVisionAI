from src.data.dataset import OralVisionDataset
from src.data.training_transforms import get_training_transforms


def main() -> None:
    dataset = OralVisionDataset(split="train")
    sample = dataset[0]

    transforms = get_training_transforms()
    patches = transforms(sample)

    print(f"Number of patches: {len(patches)}")

    for index, patch in enumerate(patches):
        image = patch["image"]
        label = patch["label"]

        print(f"\nPatch {index + 1}")
        print("Image shape:", image.shape)
        print("Label shape:", label.shape)
        print("Image min:", float(image.min()))
        print("Image max:", float(image.max()))
        print("Tumor voxels:", int((label > 0).sum()))

        assert image.shape == (1, 96, 96, 96)
        assert label.shape == (1, 96, 96, 96)

    print("\nTraining transform test passed.")


if __name__ == "__main__":
    main()
    