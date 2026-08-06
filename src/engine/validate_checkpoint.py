"""Research-use-only full-volume validation for the epoch 2 checkpoint.

This script evaluates only the validation split. It performs deterministic
preprocessing and sliding-window inference without using the expert label to
crop or otherwise guide inference.

Run from the project root::

    python -m src.engine.validate_checkpoint
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference

from src.data.cached_dataset import build_data_list
from src.data.validation_transforms import get_validation_transforms
from src.models.unet3d import create_unet3d


CHECKPOINT_PATH = Path("checkpoints/cached_epoch_002.pt")
OUTPUT_PATH = Path("outputs/validation_epoch_002.csv")
ROI_SIZE: tuple[int, int, int] = (96, 96, 96)
SW_BATCH_SIZE = 2
OVERLAP = 0.5
THRESHOLD = 0.5
LESION_CODES: tuple[str, ...] = ("AME", "DC", "KCOT", "RC")


@dataclass(frozen=True)
class ValidationResult:
    """Dice result and identifying metadata for one validation case."""

    case_id: str
    lesion_code: str
    dice: float


def _single_string(value: object, field_name: str) -> str:
    """Extract one metadata string after batch collation."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0])
    raise ValueError(
        f"Expected one {field_name} value, received {value!r}"
    )


def _check_input_tensors(
    images: torch.Tensor,
    labels: torch.Tensor,
    case_id: str,
) -> None:
    """Validate full-volume input shape, values, and binary labels."""
    if images.ndim != 5 or images.shape[:2] != (1, 1):
        raise ValueError(
            f"Unexpected image shape for {case_id}: "
            f"{tuple(images.shape)}; expected [1, 1, D, H, W]"
        )
    if labels.ndim != 5 or labels.shape[:2] != (1, 1):
        raise ValueError(
            f"Unexpected label shape for {case_id}: "
            f"{tuple(labels.shape)}; expected [1, 1, D, H, W]"
        )
    if images.shape != labels.shape:
        raise ValueError(
            f"Image/label shape mismatch for {case_id}: "
            f"{tuple(images.shape)} vs {tuple(labels.shape)}"
        )
    if not torch.isfinite(images).all():
        raise ValueError(f"Image contains non-finite values: {case_id}")
    if not torch.isfinite(labels).all():
        raise ValueError(f"Label contains non-finite values: {case_id}")

    is_binary = torch.logical_or(labels == 0, labels == 1).all()
    if not bool(is_binary.item()):
        unique_values = torch.unique(labels)
        raise ValueError(
            f"Label is not binary for {case_id}: {unique_values.tolist()}"
        )


def _load_model(device: torch.device) -> torch.nn.Module:
    """Load and validate the trained epoch 2 model checkpoint."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint: dict[str, Any] = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("epoch") != 2:
        raise ValueError(
            f"Expected epoch 2 checkpoint, got "
            f"{checkpoint.get('epoch')!r}"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint is missing model_state_dict: {CHECKPOINT_PATH}"
        )

    model = create_unet3d().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _dice_score(
    prediction: torch.Tensor,
    label: torch.Tensor,
) -> float:
    """Calculate binary Dice, treating two empty masks as a perfect match."""
    prediction_bool = prediction.to(dtype=torch.bool)
    label_bool = label.to(dtype=torch.bool)

    intersection = torch.logical_and(
        prediction_bool,
        label_bool,
    ).sum(dtype=torch.float64)
    denominator = (
        prediction_bool.sum(dtype=torch.float64)
        + label_bool.sum(dtype=torch.float64)
    )

    if denominator.item() == 0:
        return 1.0
    return float((2.0 * intersection / denominator).item())


def _print_summary(name: str, scores: list[float]) -> None:
    """Print descriptive Dice statistics for a result group."""
    if not scores:
        print(f"{name}: no validation cases")
        return

    print(
        f"{name}: n={len(scores)} | "
        f"mean={statistics.fmean(scores):.4f} | "
        f"median={statistics.median(scores):.4f} | "
        f"min={min(scores):.4f} | "
        f"max={max(scores):.4f}"
    )


def _save_results(results: list[ValidationResult]) -> None:
    """Write one CSV row per validation case."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case_id", "lesion_code", "dice"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "lesion_code": result.lesion_code,
                    "dice": f"{result.dice:.8f}",
                }
            )


def main() -> None:
    """Evaluate the epoch 2 checkpoint on the validation split."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output_device = torch.device("cpu")

    validation_data = build_data_list(split="val")
    dataset = Dataset(
        data=validation_data,
        transform=get_validation_transforms(),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    model = _load_model(device)

    print("=" * 72)
    print("OralVision AI — RESEARCH-USE-ONLY VALIDATION")
    print("=" * 72)
    print("Not for clinical diagnosis or treatment decisions.")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print("Split: val only")
    print(f"Validation cases: {len(dataset)}")
    print(f"Device: {device}")
    print(
        f"Sliding window: ROI={ROI_SIZE}, "
        f"sw_batch_size={SW_BATCH_SIZE}, overlap={OVERLAP}"
    )

    results: list[ValidationResult] = []
    start_time = perf_counter()

    with torch.inference_mode():
        for case_number, batch in enumerate(loader, start=1):
            case_id = _single_string(batch["case_id"], "case_id")
            lesion_code = _single_string(
                batch["lesion_code"],
                "lesion_code",
            )
            if lesion_code not in LESION_CODES:
                raise ValueError(
                    f"Unexpected lesion code for {case_id}: {lesion_code}"
                )

            images = batch["image"]
            labels = batch["label"]
            _check_input_tensors(images, labels, case_id)

            # Keep the full input and stitched prediction on CPU. MONAI moves
            # only each ROI batch to the accelerator, limiting GPU memory use.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=ROI_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=OVERLAP,
                    sw_device=device,
                    device=output_device,
                )

            if logits.shape != labels.shape:
                raise ValueError(
                    f"Prediction/label shape mismatch for {case_id}: "
                    f"{tuple(logits.shape)} vs {tuple(labels.shape)}"
                )
            if not torch.isfinite(logits).all():
                raise ValueError(
                    f"Model output contains non-finite values: {case_id}"
                )

            probabilities = torch.sigmoid(logits.float())
            if not torch.isfinite(probabilities).all():
                raise ValueError(
                    f"Probabilities contain non-finite values: {case_id}"
                )

            prediction = probabilities >= THRESHOLD
            dice = _dice_score(prediction, labels)
            if not torch.isfinite(torch.tensor(dice)):
                raise ValueError(f"Non-finite Dice score for {case_id}")

            results.append(
                ValidationResult(
                    case_id=case_id,
                    lesion_code=lesion_code,
                    dice=dice,
                )
            )
            print(
                f"[{case_number:03d}/{len(loader):03d}] "
                f"{case_id} ({lesion_code}) Dice: {dice:.4f}"
            )

    elapsed_seconds = perf_counter() - start_time
    all_scores = [result.dice for result in results]

    print("\n" + "=" * 72)
    print("RESEARCH-USE-ONLY VALIDATION SUMMARY")
    print("=" * 72)
    _print_summary("All validation cases", all_scores)
    for lesion_code in LESION_CODES:
        group_scores = [
            result.dice
            for result in results
            if result.lesion_code == lesion_code
        ]
        _print_summary(lesion_code, group_scores)

    _save_results(results)
    print(f"\nElapsed time: {elapsed_seconds / 60:.1f} minutes")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("Research use only — not for clinical use.")


if __name__ == "__main__":
    main()
