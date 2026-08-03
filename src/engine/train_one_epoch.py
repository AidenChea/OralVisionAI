from pathlib import Path
from time import perf_counter

import torch
from monai.data import DataLoader, list_data_collate
from monai.losses import DiceCELoss

from src.data.dataset import OralVisionDataset
from src.data.training_transforms import get_training_transforms
from src.models.unet3d import create_unet3d


CHECKPOINT_PATH = Path("checkpoints/epoch_001.pt")


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    model = create_unet3d().to(device)

    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    model.train()
    running_loss = 0.0
    start_time = perf_counter()

    print("=" * 55)
    print("OralVision AI — One Epoch Training")
    print("=" * 55)
    print(f"Device: {device}")
    print(f"Training cases: {len(dataset)}")
    print(f"Batches this epoch: {len(loader)}")

    for batch_number, batch in enumerate(loader, start=1):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            predictions = model(images)
            loss = loss_function(predictions, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at batch {batch_number}: "
                f"{loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if batch_number % 10 == 0 or batch_number == len(loader):
            average_loss = running_loss / batch_number
            elapsed_minutes = (perf_counter() - start_time) / 60

            print(
                f"Batch {batch_number:03d}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Average: {average_loss:.4f} | "
                f"Elapsed: {elapsed_minutes:.1f} min"
            )

    epoch_loss = running_loss / len(loader)
    elapsed_seconds = perf_counter() - start_time

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "training_loss": epoch_loss,
        },
        CHECKPOINT_PATH,
    )

    print("\n" + "=" * 55)
    print("Epoch completed successfully.")
    print(f"Average training loss: {epoch_loss:.4f}")
    print(
        f"Elapsed time: "
        f"{elapsed_seconds / 60:.1f} minutes"
    )
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    if device.type == "cuda":
        peak_memory_gb = (
            torch.cuda.max_memory_allocated() / 1024**3
        )
        print(
            f"Peak GPU memory allocated: "
            f"{peak_memory_gb:.2f} GB"
        )


if __name__ == "__main__":
    main()
    