"""Train OralVisionAI experiment v2 from scratch.

Experiment v2 reduces foreground overprediction through:
  - 1:3 positive/negative patch sampling (four patches per case)
  - tissue-constrained negative sampling
  - a doubled cross-entropy contribution in DiceCE loss

Full-volume validation runs after every epoch. All outputs are for research
use only and are not intended for clinical diagnosis or treatment.

Run from the project root::

    python -m src.engine.train_experiment_v2
    python -m src.engine.train_experiment_v2 --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from monai.data import DataLoader, list_data_collate
from monai.losses import DiceCELoss

from src.data.cached_dataset import OralVisionCachedDataset
from src.data.cached_training_transforms_v2 import (
    DEFAULT_CACHE_DIR_V2,
    PATCH_SIZE,
    get_cached_training_transforms_v2,
)
from src.engine.validate_experiment_v2 import validate_model
from src.models.unet3d import create_unet3d


DEFAULT_NUM_EPOCHS = 5
CHECKPOINT_DIR = Path("checkpoints/experiment_v2")
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"
TRAINING_CACHE_DIR = Path(DEFAULT_CACHE_DIR_V2)
EXPECTED_BATCH_SHAPE = (4, 1, *PATCH_SIZE)


def _check_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_number: int,
) -> None:
    """Verify the four sampled patches and their binary labels."""
    if tuple(images.shape) != EXPECTED_BATCH_SHAPE:
        raise ValueError(
            f"Unexpected image shape at batch {batch_number}: "
            f"{tuple(images.shape)}, expected {EXPECTED_BATCH_SHAPE}"
        )
    if tuple(labels.shape) != EXPECTED_BATCH_SHAPE:
        raise ValueError(
            f"Unexpected label shape at batch {batch_number}: "
            f"{tuple(labels.shape)}, expected {EXPECTED_BATCH_SHAPE}"
        )
    if not bool(torch.isfinite(images).all().item()):
        raise ValueError(
            f"Non-finite images at batch {batch_number}"
        )
    if not bool(torch.isfinite(labels).all().item()):
        raise ValueError(
            f"Non-finite labels at batch {batch_number}"
        )
    if not bool(torch.logical_or(labels == 0, labels == 1).all().item()):
        raise ValueError(
            f"Non-binary labels at batch {batch_number}"
        )


def _build_checkpoint(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    training_loss: float,
    training_seconds: float,
    mean_validation_dice: float,
    predicted_foreground_percent: float,
    true_foreground_percent: float,
) -> dict[str, object]:
    """Create a complete, resumable experiment checkpoint."""
    return {
        "experiment": "v2_foreground_overprediction",
        "research_use_only": True,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "training_loss": training_loss,
        "training_seconds": training_seconds,
        "mean_validation_dice": mean_validation_dice,
        "mean_predicted_foreground_percent": (
            predicted_foreground_percent
        ),
        "mean_true_foreground_percent": true_foreground_percent,
        "cache_dir": str(TRAINING_CACHE_DIR.resolve()),
        "loss_settings": {
            "lambda_dice": 1.0,
            "lambda_ce": 2.0,
            "sigmoid": True,
            "squared_pred": True,
        },
        "sampling_settings": {
            "patch_size": PATCH_SIZE,
            "pos": 1,
            "neg": 3,
            "num_samples": 4,
        },
    }


def main() -> None:
    """Train a fresh model with validation after every epoch."""
    parser = argparse.ArgumentParser(
        description=(
            "Train research-use-only OralVisionAI experiment v2."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=(
            "Number of fresh-model training epochs "
            f"(default: {DEFAULT_NUM_EPOCHS})."
        ),
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    num_epochs: int = args.epochs

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = OralVisionCachedDataset(
        split="train",
        cache_dir=TRAINING_CACHE_DIR,
        transform=get_cached_training_transforms_v2(),
    )
    if dataset.cache_dir.resolve() != TRAINING_CACHE_DIR.resolve():
        raise ValueError(
            f"Unexpected cache directory: {dataset.cache_dir.resolve()}"
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # Intentionally initialize a fresh model. No prior checkpoint is loaded.
    model = create_unet3d().to(device)
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
        lambda_dice=1.0,
        lambda_ce=2.0,
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
    best_mean_dice = float("-inf")
    best_epoch = 0

    print("=" * 72)
    print("OralVision AI — EXPERIMENT V2 TRAINING")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print(f"Device: {device}")
    print("Initialization: fresh model (no checkpoint loaded)")
    print(f"Epochs: {num_epochs}")
    print(f"Training cases: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")
    print(f"Patch batch shape: {EXPECTED_BATCH_SHAPE}")
    print("Sampling ratio: pos=1, neg=3, num_samples=4")
    print("Loss weights: lambda_dice=1.0, lambda_ce=2.0")
    print(f"Persistent cache: {dataset.cache_dir.resolve()}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = perf_counter()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        print("\n" + "=" * 72)
        print(
            f"EPOCH {epoch}/{num_epochs} — "
            "RESEARCH-USE-ONLY TRAINING"
        )
        print("=" * 72)

        for batch_number, batch in enumerate(loader, start=1):
            images = batch["image"]
            labels = batch["label"]
            _check_training_batch(
                images,
                labels,
                batch_number,
            )

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                predictions = model(images)
                loss = loss_function(predictions, labels)

            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, "
                    f"batch {batch_number}: {loss.item()}"
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (
                batch_number % 10 == 0
                or batch_number == len(loader)
            ):
                average_loss = running_loss / batch_number
                elapsed_minutes = (
                    perf_counter() - epoch_start
                ) / 60
                print(
                    f"Batch {batch_number:03d}/{len(loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Average: {average_loss:.4f} | "
                    f"Elapsed: {elapsed_minutes:.1f} min"
                )

        epoch_seconds = perf_counter() - epoch_start
        epoch_loss = running_loss / len(loader)
        average_batch_seconds = epoch_seconds / len(loader)

        print("\nTraining epoch summary")
        print(f"Epoch: {epoch}/{num_epochs}")
        print(f"Average training loss: {epoch_loss:.4f}")
        print(
            f"Training epoch time: {epoch_seconds / 60:.1f} min "
            f"({epoch_seconds:.1f} s)"
        )
        print(
            f"Average batch time: {average_batch_seconds:.3f} s"
        )
        if device.type == "cuda":
            peak_memory_gb = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
            print(
                f"Peak GPU memory allocated: "
                f"{peak_memory_gb:.2f} GB"
            )

        validation = validate_model(
            model=model,
            device=device,
            epoch=epoch,
        )

        checkpoint = _build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            training_loss=epoch_loss,
            training_seconds=epoch_seconds,
            mean_validation_dice=validation.mean_dice,
            predicted_foreground_percent=(
                validation.mean_predicted_foreground_percent
            ),
            true_foreground_percent=(
                validation.mean_true_foreground_percent
            ),
        )
        epoch_checkpoint_path = (
            CHECKPOINT_DIR
            / f"epoch_{epoch:03d}.pt"
        )
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"Epoch checkpoint saved: {epoch_checkpoint_path}")

        if validation.mean_dice > best_mean_dice:
            best_mean_dice = validation.mean_dice
            best_epoch = epoch
            torch.save(checkpoint, BEST_CHECKPOINT_PATH)
            print(
                f"New best checkpoint saved: {BEST_CHECKPOINT_PATH} "
                f"(mean Dice={best_mean_dice:.4f})"
            )
        else:
            print(
                f"Best checkpoint unchanged: epoch {best_epoch}, "
                f"mean Dice={best_mean_dice:.4f}"
            )

    print("\n" + "=" * 72)
    print("EXPERIMENT V2 COMPLETED")
    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(f"Best mean validation Dice: {best_mean_dice:.4f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
