import torch
from monai.data import DataLoader, list_data_collate

from src.data.dataset import OralVisionDataset
from src.data.training_transforms import get_training_transforms


def main() -> None:
    transforms = get_training_transforms()

    dataset = OralVisionDataset(
        split="train",
        transform=transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    print("=" * 50)
    print("OralVision AI — DataLoader Test")
    print("=" * 50)
    print(f"Dataset cases: {len(dataset)}")
    print(f"Number of loader batches: {len(loader)}")

    batch = next(iter(loader))

    images = batch["image"]
    labels = batch["label"]

    print("\nBatch information")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    print("Image minimum:", float(images.min()))
    print("Image maximum:", float(images.max()))
    print("Tumor voxels:", int((labels > 0).sum()))

    # One case produces two sampled patches.
    assert images.shape == (2, 1, 96, 96, 96)
    assert labels.shape == (2, 1, 96, 96, 96)

    if torch.cuda.is_available():
        images = images.to("cuda", non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)

        print("\nGPU transfer")
        print("Image device:", images.device)
        print("Label device:", labels.device)

        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU memory allocated: {allocated_gb:.3f} GB")

    print("\nDataLoader test passed.")


if __name__ == "__main__":
    main()
    