"""Train the OralVisionAI coarse lesion localizer (Experiment V4 stage 1).

Trains a lightweight 3D U-Net on full downsampled 128^3 CBCT volumes using
DiceCE loss and mixed precision. Validation metrics emphasize whether the
predicted lesion center is accurate enough to crop for fine segmentation.

Run from the project root::

    python -m src.engine.train_localizer
    python -m src.engine.train_localizer --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.data.utils import pickle_hashing
from monai.losses import DiceCELoss

from src.data.cached_dataset import build_data_list
from src.data.localization_transforms import (
    LOCALIZER_CACHE_DIR,
    LOCALIZER_SPATIAL_SIZE,
    get_localization_transforms,
)
from src.engine.validate_localizer import (
    CHECKPOINT_DIR,
    validate_localizer,
)
from src.models.localizer3d import create_localizer3d


DEFAULT_EPOCHS = 20
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
EXPECTED_BATCH_SHAPE = (1, 1, *LOCALIZER_SPATIAL_SIZE)


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


def _build_checkpoint(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    training_loss: float,
    training_seconds: float,
    mean_coarse_dice: float,
    mean_center_distance_mm: float,
    center_inside_lesion_rate: float,
    crop_overlap_rate: float,
    lesion_recall: float,
) -> dict[str, object]:
    return {
        "experiment": "localizer_v1",
        "research_use_only": True,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "training_loss": training_loss,
        "training_seconds": training_seconds,
        "mean_coarse_dice": mean_coarse_dice,
        "mean_center_distance_mm": mean_center_distance_mm,
        "center_inside_lesion_rate": center_inside_lesion_rate,
        "crop_overlap_rate": crop_overlap_rate,
        "lesion_recall": lesion_recall,
        "cache_dir": str(Path(LOCALIZER_CACHE_DIR).resolve()),
        "spatial_size": LOCALIZER_SPATIAL_SIZE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train research-use-only OralVisionAI localizer V1."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transform = get_localization_transforms()
    dataset = PersistentDataset(
        data=build_data_list(split="train"),
        transform=transform,
        cache_dir=LOCALIZER_CACHE_DIR,
        hash_transform=pickle_hashing,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_localizer3d().to(device)
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
        lambda_dice=1.0,
        lambda_ce=1.0,
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
    best_score = float("-inf")
    best_epoch = 0

    print("=" * 72)
    print("OralVision AI — LOCALIZER V1 TRAINING")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print(f"Device: {device}")
    print("Initialization: fresh lightweight 3D U-Net")
    print(f"Epochs: {args.epochs}")
    print(f"Training cases: {len(dataset)}")
    print(f"Input shape: {EXPECTED_BATCH_SHAPE}")
    print(f"Cache: {Path(LOCALIZER_CACHE_DIR).resolve()}")
    print("Objective: DiceCE (lambda_dice=1.0, lambda_ce=1.0)")

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

            if batch_number % 20 == 0 or batch_number == len(loader):
                print(
                    f"Batch {batch_number:03d}/{len(loader)} | "
                    f"Loss={loss.item():.4f} | "
                    f"Average={running_loss / batch_number:.4f}"
                )

        elapsed = perf_counter() - start
        epoch_loss = running_loss / len(loader)
        print(f"Epoch loss: {epoch_loss:.4f}")
        print(f"Epoch runtime: {elapsed / 60:.1f} minutes")
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory: {peak_gb:.2f} GB")

        validation = validate_localizer(model, device, epoch)

        # Prefer crop-overlap success, then coarse Dice, when selecting best.
        selection_score = (
            validation.crop_overlap_rate
            + 0.25 * validation.mean_coarse_dice
        )
        state = _build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            training_loss=epoch_loss,
            training_seconds=elapsed,
            mean_coarse_dice=validation.mean_coarse_dice,
            mean_center_distance_mm=validation.mean_center_distance_mm,
            center_inside_lesion_rate=(
                validation.center_inside_lesion_rate
            ),
            crop_overlap_rate=validation.crop_overlap_rate,
            lesion_recall=validation.lesion_recall,
        )
        epoch_path = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
        torch.save(state, epoch_path)
        print(f"Checkpoint saved: {epoch_path}")

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            torch.save(state, BEST_CHECKPOINT)
            print(
                f"New best checkpoint: {BEST_CHECKPOINT} "
                f"(crop overlap="
                f"{100.0 * validation.crop_overlap_rate:.1f}%, "
                f"Dice={validation.mean_coarse_dice:.4f})"
            )
        else:
            print(
                f"Best preserved: epoch {best_epoch}, "
                f"selection_score={best_score:.4f}"
            )

    print("\nLocalizer V1 training completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {BEST_CHECKPOINT.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
