"""Train one epoch using the persistent-cache data pipeline.

Run from the project root::

    python -m src.engine.train_cached_one_epoch
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch
from monai.data import DataLoader, list_data_collate
from monai.losses import DiceCELoss

from src.data.cached_dataset import OralVisionCachedDataset
from src.data.cached_training_transforms import (
    DEFAULT_CACHE_DIR,
    get_cached_training_transforms,
)
from src.models.unet3d import create_unet3d


CHECKPOINT_PATH = Path("checkpoints/cached_epoch_001.pt")
EXPECTED_CACHE_DIR = Path(DEFAULT_CACHE_DIR)


def _verify_cache_directory(cache_dir: Path) -> None:
    """Ensure the dataset uses the expected persistent cache location."""
    resolved_cache_dir = cache_dir.resolve()
    expected_cache_dir = EXPECTED_CACHE_DIR.resolve()

    if resolved_cache_dir != expected_cache_dir:
        raise ValueError(
            f"Unexpected cache directory: {resolved_cache_dir}. "
            f"Expected: {expected_cache_dir}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.is_dir():
        raise NotADirectoryError(
            f"Cache path is not a directory: {cache_dir}"
        )


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transform = get_cached_training_transforms()
    dataset = OralVisionCachedDataset(
        split="train",
        cache_dir=EXPECTED_CACHE_DIR,
        transform=transform,
    )

    _verify_cache_directory(dataset.cache_dir)

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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("=" * 55)
    print("OralVision AI — Cached One Epoch Training")
    print("=" * 55)
    print(f"Device: {device}")
    print(f"Cache directory: {dataset.cache_dir.resolve()}")
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

    batch_count = len(loader)
    epoch_loss = running_loss / batch_count
    elapsed_seconds = perf_counter() - start_time
    average_batch_seconds = elapsed_seconds / batch_count

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
            "cache_dir": str(dataset.cache_dir.resolve()),
        },
        CHECKPOINT_PATH,
    )

    print("\n" + "=" * 55)
    print("Epoch completed successfully.")
    print(f"Average training loss: {epoch_loss:.4f}")
    print(
        f"Total epoch time: "
        f"{elapsed_seconds / 60:.1f} minutes "
        f"({elapsed_seconds:.1f} seconds)"
    )
    print(
        f"Average time per batch: {average_batch_seconds:.3f} seconds"
    )
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Cache directory verified: {dataset.cache_dir.resolve()}")

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
