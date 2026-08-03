import torch
from monai.data import DataLoader, list_data_collate

from src.data.dataset import OralVisionDataset
from src.data.training_transforms import get_training_transforms
from src.models.unet3d import create_unet3d


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OralVisionDataset(
        split="train",
        transform=get_training_transforms(),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))
    images = batch["image"].to(device)

    model = create_unet3d().to(device)
    model.eval()

    print("=" * 50)
    print("OralVision AI — 3D U-Net Forward Test")
    print("=" * 50)
    print("Device:", device)
    print("Input shape:", images.shape)

    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(images)

    print("Output shape:", outputs.shape)
    print("Output minimum:", float(outputs.min()))
    print("Output maximum:", float(outputs.max()))

    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.memory_reserved() / (1024**3)

    print(f"GPU memory allocated: {allocated_gb:.2f} GB")
    print(f"GPU memory reserved: {reserved_gb:.2f} GB")

    assert outputs.shape == images.shape

    print("\n3D U-Net forward pass succeeded.")


if __name__ == "__main__":
    main()
    