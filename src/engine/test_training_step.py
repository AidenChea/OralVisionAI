import torch
from monai.data import DataLoader, list_data_collate
from monai.losses import DiceCELoss

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
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    model = create_unet3d().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    print("=" * 50)
    print("OralVision AI — Training Step Test")
    print("=" * 50)
    print("Device:", device)
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        outputs = model(images)
        loss = loss_function(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Loss: {loss.item():.4f}")

    if device.type == "cuda":
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

        print(f"GPU memory allocated: {allocated_gb:.2f} GB")
        print(f"GPU memory reserved: {reserved_gb:.2f} GB")
        print(f"Peak GPU memory allocated: {peak_gb:.2f} GB")

    print("\nTraining step succeeded.")


if __name__ == "__main__":
    main()
    