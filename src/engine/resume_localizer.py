"""Resume OralVisionAI localizer training from an existing checkpoint.

Research use only. Not for clinical diagnosis or treatment decisions.

Example from the project root::

    python -m src.engine.resume_localizer \
        --checkpoint checkpoints/localizer_v1/epoch_001.pt \
        --additional-epochs 9
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

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
from src.engine.train_localizer import _build_checkpoint, _check_batch
from src.engine.validate_localizer import (
    BEST_CHECKPOINT,
    CHECKPOINT_DIR,
    validate_localizer,
)
from src.models.localizer3d import create_localizer3d


DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "epoch_001.pt"
DEFAULT_ADDITIONAL_EPOCHS = 9


def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "epoch",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing keys: {missing}"
        )

    return checkpoint


def _selection_score(checkpoint: dict[str, Any]) -> float:
    crop_overlap = float(checkpoint.get("crop_overlap_rate", 0.0))
    coarse_dice = float(checkpoint.get("mean_coarse_dice", 0.0))
    return crop_overlap + 0.25 * coarse_dice


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume research-use-only OralVisionAI localizer training."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Checkpoint to resume from "
            f"(default: {DEFAULT_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=DEFAULT_ADDITIONAL_EPOCHS,
        help=(
            "Number of additional epochs to train "
            f"(default: {DEFAULT_ADDITIONAL_EPOCHS})."
        ),
    )
    args = parser.parse_args()

    if args.additional_epochs < 1:
        parser.error("--additional-epochs must be at least 1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = _load_checkpoint(args.checkpoint, device)
    restored_epoch = int(checkpoint["epoch"])
    first_epoch = restored_epoch + 1
    final_epoch = restored_epoch + args.additional_epochs

    dataset = PersistentDataset(
        data=build_data_list(split="train"),
        transform=get_localization_transforms(),
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
    model.load_state_dict(checkpoint["model_state_dict"])

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
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_epoch = restored_epoch
    best_score = _selection_score(checkpoint)

    if BEST_CHECKPOINT.is_file():
        best_checkpoint: dict[str, Any] = torch.load(
            BEST_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
        candidate_score = _selection_score(best_checkpoint)
        if candidate_score >= best_score:
            best_score = candidate_score
            best_epoch = int(best_checkpoint.get("epoch", restored_epoch))

    print("=" * 72)
    print("OralVision AI — RESUME LOCALIZER V1 TRAINING")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Checkpoint restored: {args.checkpoint.resolve()}")
    print(f"Restored epoch: {restored_epoch}")
    print(f"Continuing with epochs {first_epoch} through {final_epoch}")
    print(f"Training cases: {len(dataset)}")
    print(f"Input shape: {(1, 1, *LOCALIZER_SPATIAL_SIZE)}")
    print(f"Cache: {Path(LOCALIZER_CACHE_DIR).resolve()}")
    print(f"Existing best epoch: {best_epoch}")

    for epoch in range(first_epoch, final_epoch + 1):
        model.train()
        running_loss = 0.0
        start = perf_counter()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        print(f"\nEpoch {epoch}/{final_epoch}")

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

    print("\nLocalizer resume training completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {BEST_CHECKPOINT.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
