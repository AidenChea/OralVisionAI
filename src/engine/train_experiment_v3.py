"""Train research-use-only OralVisionAI Experiment V3 from scratch."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from monai.data import DataLoader, list_data_collate
from monai.losses import DiceCELoss

from src.data.cached_dataset import OralVisionCachedDataset
from src.data.cached_training_transforms_v3 import (
    DEFAULT_CACHE_DIR_V3,
    PATCH_SIZE,
    get_cached_training_transforms_v3,
)
from src.engine.validate_experiment_v3 import validate_model
from src.models.unet3d import create_unet3d


DEFAULT_EPOCHS = 5
CHECKPOINT_DIR = Path("checkpoints/experiment_v3")
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
TRAINING_CACHE = Path(DEFAULT_CACHE_DIR_V3)
EXPECTED_BATCH_SHAPE = (8, 1, *PATCH_SIZE)


def _check_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_number: int,
) -> None:
    if tuple(images.shape) != EXPECTED_BATCH_SHAPE:
        raise ValueError(
            f"Unexpected images at batch {batch_number}: "
            f"{tuple(images.shape)}"
        )
    if tuple(labels.shape) != EXPECTED_BATCH_SHAPE:
        raise ValueError(
            f"Unexpected labels at batch {batch_number}: "
            f"{tuple(labels.shape)}"
        )
    if not bool(torch.isfinite(images).all().item()):
        raise ValueError(f"Non-finite images at batch {batch_number}")
    if not bool(torch.isfinite(labels).all().item()):
        raise ValueError(f"Non-finite labels at batch {batch_number}")
    if not bool(torch.logical_or(labels == 0, labels == 1).all().item()):
        raise ValueError(f"Non-binary labels at batch {batch_number}")


def _checkpoint(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    training_loss: float,
    training_seconds: float,
    mean_dice: float,
    predicted_foreground: float,
    true_foreground: float,
) -> dict[str, object]:
    return {
        "experiment": "v3_prevalence_mismatch",
        "research_use_only": True,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "training_loss": training_loss,
        "training_seconds": training_seconds,
        "mean_validation_dice": mean_dice,
        "mean_predicted_foreground_percent": predicted_foreground,
        "mean_true_foreground_percent": true_foreground,
        "cache_dir": str(TRAINING_CACHE.resolve()),
        "loss_settings": {
            "lambda_dice": 1.0,
            "lambda_ce": 3.0,
        },
        "sampling_settings": {
            "patch_size": PATCH_SIZE,
            "pos": 1,
            "neg": 7,
            "num_samples": 8,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train research-use-only OralVisionAI Experiment V3."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS}).",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dataset = OralVisionCachedDataset(
        split="train",
        cache_dir=TRAINING_CACHE,
        transform=get_cached_training_transforms_v3(),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # V3 always begins from a fresh model.
    model = create_unet3d().to(device)
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
        lambda_dice=1.0,
        lambda_ce=3.0,
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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_dice = float("-inf")
    best_epoch = 0

    print("=" * 72)
    print("OralVision AI — EXPERIMENT V3 TRAINING")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print("Initialization: fresh 3D U-Net")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Training cases: {len(dataset)}")
    print(f"Patch batch shape: {EXPECTED_BATCH_SHAPE}")
    print("Sampling: pos=1, neg=7, num_samples=8")
    print("Loss: lambda_dice=1.0, lambda_ce=3.0")
    print(f"Cache: {dataset.cache_dir.resolve()}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start = perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        print(f"\nEpoch {epoch}/{args.epochs}")
        for batch_number, batch in enumerate(loader, start=1):
            images = batch["image"]
            labels = batch["label"]
            _check_batch(images, labels, batch_number)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = loss_function(logits, labels)

            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, "
                    f"batch {batch_number}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if batch_number % 10 == 0 or batch_number == len(loader):
                print(
                    f"Batch {batch_number:03d}/{len(loader)} | "
                    f"Loss={loss.item():.4f} | "
                    f"Average={running_loss / batch_number:.4f}"
                )

        elapsed = perf_counter() - start
        epoch_loss = running_loss / len(loader)
        print(f"Epoch loss: {epoch_loss:.4f}")
        print(
            f"Epoch runtime: {elapsed / 60:.1f} min "
            f"({elapsed:.1f} s)"
        )
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory: {peak_gb:.2f} GB")

        validation = validate_model(model, device, epoch)
        state = _checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            training_loss=epoch_loss,
            training_seconds=elapsed,
            mean_dice=validation.mean_dice,
            predicted_foreground=(
                validation.mean_predicted_foreground_percent
            ),
            true_foreground=(
                validation.mean_true_foreground_percent
            ),
        )
        epoch_path = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
        torch.save(state, epoch_path)
        print(f"Checkpoint saved: {epoch_path}")

        if validation.mean_dice > best_dice:
            best_dice = validation.mean_dice
            best_epoch = epoch
            torch.save(state, BEST_CHECKPOINT)
            print(
                f"New best checkpoint: {BEST_CHECKPOINT} "
                f"(mean Dice={best_dice:.4f})"
            )

    print("\nExperiment V3 completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best mean validation Dice: {best_dice:.4f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
