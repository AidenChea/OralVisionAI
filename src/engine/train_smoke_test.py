from pathlib import Path

import torch
from monai.data import DataLoader, list_data_collate
from monai.losses import DiceCELoss

from src.data.dataset import OralVisionDataset
from src.data.training_transforms import get_training_transforms
from src.models.unet3d import create_unet3d


CHECKPOINT_PATH = Path("checkpoints/smoke_test_model.pt")
MAX_BATCHES = 10


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

    print("=" * 55)
    print("OralVision AI — Training Smoke Test")
    print("=" * 55)
    print(f"Device: {device}")
    print(f"Training cases: {len(dataset)}")
    print(f"Maximum batches: {MAX_BATCHES}")

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
                f"Non-finite loss detected: {loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        average_loss = running_loss / batch_number

        print(
            f"Batch {batch_number:02d}/{MAX_BATCHES} | "
            f"Loss: {loss.item():.4f} | "
            f"Average: {average_loss:.4f}"
        )

        if batch_number >= MAX_BATCHES:
            break

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "average_loss": running_loss / MAX_BATCHES,
            "batches_completed": MAX_BATCHES,
        },
        CHECKPOINT_PATH,
    )

    print("\n" + "=" * 55)
    print("Smoke test completed successfully.")
    print(
        f"Final average loss: "
        f"{running_loss / MAX_BATCHES:.4f}"
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
    