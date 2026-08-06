"""Resume OralVisionAI Experiment V3 from its epoch-1 checkpoint.

The populated V3 training cache is required and is never cleared by this
script. With the default arguments, epochs 2 through 5 are trained,
validated, and saved.

Run from the project root::

    python -m src.engine.resume_experiment_v3
    python -m src.engine.resume_experiment_v3 --additional-epochs 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from monai.data import DataLoader, list_data_collate
from monai.losses import DiceCELoss

from src.data.cached_dataset import (
    OralVisionCachedDataset,
    load_split_cases,
)
from src.data.cached_training_transforms_v3 import (
    get_cached_training_transforms_v3,
)
from src.engine.train_experiment_v3 import (
    BEST_CHECKPOINT,
    CHECKPOINT_DIR,
    TRAINING_CACHE,
    _check_batch,
    _checkpoint,
)
from src.engine.validate_experiment_v3 import validate_model
from src.models.unet3d import create_unet3d


SOURCE_CHECKPOINT = CHECKPOINT_DIR / "epoch_001.pt"
DEFAULT_ADDITIONAL_EPOCHS = 4


@dataclass(frozen=True)
class TrendRow:
    """Compact epoch metrics printed after resumed training."""

    epoch: int
    training_loss: float
    training_minutes: float
    mean_dice: float
    predicted_foreground_percent: float
    true_foreground_percent: float


def _load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load and validate one Experiment V3 checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("experiment") != "v3_prevalence_mismatch":
        raise ValueError(f"Not an Experiment V3 checkpoint: {path}")

    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "training_loss",
        "mean_validation_dice",
        "mean_predicted_foreground_percent",
        "mean_true_foreground_percent",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise KeyError(
            f"Checkpoint {path} is missing keys: {sorted(missing)}"
        )
    return checkpoint


def _verify_complete_training_cache() -> int:
    """Refuse to run if V3 cache entries would need rebuilding."""
    if not TRAINING_CACHE.is_dir():
        raise FileNotFoundError(
            f"V3 cache directory is missing: {TRAINING_CACHE}. "
            "Refusing to rebuild it in the resume script."
        )

    expected_cases = len(load_split_cases("train"))
    cached_files = list(TRAINING_CACHE.glob("*.pt"))
    if len(cached_files) < expected_cases:
        raise RuntimeError(
            f"V3 cache is incomplete: found {len(cached_files)} files "
            f"for {expected_cases} training cases. Refusing to rebuild."
        )
    return len(cached_files)


def _trend_from_checkpoint(
    checkpoint: dict[str, Any],
) -> TrendRow:
    """Convert stored checkpoint metrics into one trend row."""
    return TrendRow(
        epoch=int(checkpoint["epoch"]),
        training_loss=float(checkpoint["training_loss"]),
        training_minutes=float(
            checkpoint.get("training_seconds", 0.0)
        )
        / 60.0,
        mean_dice=float(checkpoint["mean_validation_dice"]),
        predicted_foreground_percent=float(
            checkpoint["mean_predicted_foreground_percent"]
        ),
        true_foreground_percent=float(
            checkpoint["mean_true_foreground_percent"]
        ),
    )


def _print_trend(rows: list[TrendRow]) -> None:
    """Print compact fixed-width metrics without external dependencies."""
    print("\n" + "=" * 84)
    print("EXPERIMENT V3 VALIDATION TREND — RESEARCH USE ONLY")
    print("=" * 84)
    print(
        f"{'Epoch':>5}  {'Train loss':>10}  {'Train min':>9}  "
        f"{'Mean Dice':>10}  {'Pred FG %':>10}  {'True FG %':>10}"
    )
    print("-" * 84)
    for row in rows:
        print(
            f"{row.epoch:>5d}  "
            f"{row.training_loss:>10.4f}  "
            f"{row.training_minutes:>9.1f}  "
            f"{row.mean_dice:>10.4f}  "
            f"{row.predicted_foreground_percent:>10.4f}  "
            f"{row.true_foreground_percent:>10.4f}"
        )


def main() -> None:
    """Restore epoch 1 and train the requested additional epochs."""
    parser = argparse.ArgumentParser(
        description=(
            "Resume research-use-only OralVisionAI Experiment V3."
        )
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=DEFAULT_ADDITIONAL_EPOCHS,
        help=(
            "Epochs to train after epoch 1 "
            f"(default: {DEFAULT_ADDITIONAL_EPOCHS})."
        ),
    )
    args = parser.parse_args()
    if args.additional_epochs < 1:
        parser.error("--additional-epochs must be at least 1")

    cache_file_count = _verify_complete_training_cache()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    source = _load_checkpoint(SOURCE_CHECKPOINT, device)
    if int(source["epoch"]) != 1:
        raise ValueError(
            f"Expected epoch 1 in {SOURCE_CHECKPOINT}, "
            f"got {source['epoch']!r}"
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

    model.load_state_dict(source["model_state_dict"])
    optimizer.load_state_dict(source["optimizer_state_dict"])
    scaler.load_state_dict(source["scaler_state_dict"])
    source_trend = _trend_from_checkpoint(source)

    if BEST_CHECKPOINT.is_file():
        # Only metadata is needed from the best checkpoint; loading it on CPU
        # avoids retaining a second model state in GPU memory.
        current_best = _load_checkpoint(
            BEST_CHECKPOINT,
            torch.device("cpu"),
        )
        best_dice = float(current_best["mean_validation_dice"])
        best_epoch = int(current_best["epoch"])
        del current_best
    else:
        # No prior best file exists. Establish epoch 1 as the baseline.
        best_dice = float(source["mean_validation_dice"])
        best_epoch = 1
        torch.save(source, BEST_CHECKPOINT)

    first_epoch = 2
    final_epoch = 1 + args.additional_epochs
    trend = [source_trend]
    # Model/optimizer/scaler now own the restored state. Release the loaded
    # checkpoint container before training to avoid duplicate tensor storage.
    del source

    print("=" * 72)
    print("OralVision AI — RESUME EXPERIMENT V3")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print(f"Device: {device}")
    print(
        f"Epoch 1 restored successfully from: "
        f"{SOURCE_CHECKPOINT.resolve()}"
    )
    print("Restored model, AdamW optimizer, and GradScaler states.")
    print(
        f"Continuing with epochs {first_epoch} through {final_epoch}"
    )
    print(
        f"Existing best: epoch {best_epoch}, "
        f"mean Dice={best_dice:.4f}"
    )
    print(
        f"V3 cache verified: {TRAINING_CACHE.resolve()} "
        f"({cache_file_count} files); cache will not be cleared."
    )

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

            if batch_number % 10 == 0 or batch_number == len(loader):
                print(
                    f"Batch {batch_number:03d}/{len(loader)} | "
                    f"Loss={loss.item():.4f} | "
                    f"Average={running_loss / batch_number:.4f}"
                )

        elapsed = perf_counter() - start
        training_loss = running_loss / len(loader)
        print(f"Epoch loss: {training_loss:.4f}")
        print(f"Epoch runtime: {elapsed / 60:.1f} minutes")
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory: {peak_gb:.2f} GB")

        validation = validate_model(model, device, epoch)
        state = _checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            training_loss=training_loss,
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
                f"New best checkpoint: epoch {best_epoch}, "
                f"mean Dice={best_dice:.4f}"
            )
        else:
            print(
                f"Best preserved: epoch {best_epoch}, "
                f"mean Dice={best_dice:.4f}"
            )

        trend.append(
            TrendRow(
                epoch=epoch,
                training_loss=training_loss,
                training_minutes=elapsed / 60.0,
                mean_dice=validation.mean_dice,
                predicted_foreground_percent=(
                    validation.mean_predicted_foreground_percent
                ),
                true_foreground_percent=(
                    validation.mean_true_foreground_percent
                ),
            )
        )

    _print_trend(trend)
    print(
        f"\nBest checkpoint: epoch {best_epoch}, "
        f"mean Dice={best_dice:.4f}"
    )
    print(f"Path: {BEST_CHECKPOINT.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
