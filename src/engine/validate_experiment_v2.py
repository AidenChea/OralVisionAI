"""Research-use-only validation for OralVisionAI experiment v2.

This module exposes ``validate_model`` for validation after each training
epoch and can also evaluate a saved v2 checkpoint directly.

Run from the project root::

    python -m src.engine.validate_experiment_v2
    python -m src.engine.validate_experiment_v2 --checkpoint checkpoints/experiment_v2/epoch_003.pt
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.data.utils import pickle_hashing
from monai.inferers import sliding_window_inference

from src.data.cached_dataset import build_data_list
from src.data.validation_transforms import get_validation_transforms
from src.models.unet3d import create_unet3d


ROI_SIZE: tuple[int, int, int] = (96, 96, 96)
SW_BATCH_SIZE = 2
OVERLAP = 0.5
PREDICTION_THRESHOLD = 0.5
OUTPUT_DIR = Path("outputs/experiment_v2")
VALIDATION_CACHE_DIR = Path("data/cache/validation_v2")
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/experiment_v2/best.pt")


@dataclass(frozen=True)
class ValidationCaseResult:
    """Metrics for one full validation volume."""

    case_id: str
    lesion_code: str
    dice: float
    predicted_foreground_percent: float
    true_foreground_percent: float


@dataclass(frozen=True)
class ValidationSummary:
    """Aggregate metrics returned to the training experiment."""

    epoch: int
    mean_dice: float
    mean_predicted_foreground_percent: float
    mean_true_foreground_percent: float
    csv_path: Path
    case_results: tuple[ValidationCaseResult, ...]


def _single_string(value: object, name: str) -> str:
    """Extract a single metadata string after DataLoader collation."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0])
    raise ValueError(f"Expected one {name}, received {value!r}")


def _check_inputs(
    images: torch.Tensor,
    labels: torch.Tensor,
    case_id: str,
) -> None:
    """Check full-volume shape, finite values, and binary labels."""
    expected_prefix = (1, 1)
    if images.ndim != 5 or tuple(images.shape[:2]) != expected_prefix:
        raise ValueError(
            f"Unexpected image shape for {case_id}: "
            f"{tuple(images.shape)}"
        )
    if labels.ndim != 5 or tuple(labels.shape[:2]) != expected_prefix:
        raise ValueError(
            f"Unexpected label shape for {case_id}: "
            f"{tuple(labels.shape)}"
        )
    if images.shape != labels.shape:
        raise ValueError(
            f"Image/label shape mismatch for {case_id}: "
            f"{tuple(images.shape)} vs {tuple(labels.shape)}"
        )
    if not bool(torch.isfinite(images).all().item()):
        raise ValueError(f"Non-finite image values for {case_id}")
    if not bool(torch.isfinite(labels).all().item()):
        raise ValueError(f"Non-finite label values for {case_id}")
    if not bool(torch.logical_or(labels == 0, labels == 1).all().item()):
        raise ValueError(
            f"Non-binary validation label for {case_id}: "
            f"{torch.unique(labels).tolist()}"
        )


def _dice_score(
    prediction: torch.Tensor,
    label: torch.Tensor,
) -> float:
    """Compute binary Dice; two empty masks receive Dice 1."""
    prediction = prediction.to(dtype=torch.bool)
    label = label.to(dtype=torch.bool)
    intersection = torch.logical_and(prediction, label).sum(
        dtype=torch.float64
    )
    denominator = (
        prediction.sum(dtype=torch.float64)
        + label.sum(dtype=torch.float64)
    )
    if denominator.item() == 0:
        return 1.0
    return float((2.0 * intersection / denominator).item())


def _save_csv(
    results: list[ValidationCaseResult],
    epoch: int,
    output_dir: Path,
) -> Path:
    """Save per-case validation metrics for one epoch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"validation_epoch_{epoch:03d}.csv"
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "lesion_code",
                "dice",
                "predicted_foreground_percent",
                "true_foreground_percent",
                "research_use_only",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "lesion_code": result.lesion_code,
                    "dice": f"{result.dice:.8f}",
                    "predicted_foreground_percent": (
                        f"{result.predicted_foreground_percent:.8f}"
                    ),
                    "true_foreground_percent": (
                        f"{result.true_foreground_percent:.8f}"
                    ),
                    "research_use_only": "true",
                }
            )
    return csv_path


def validate_model(
    model: torch.nn.Module,
    device: torch.device,
    epoch: int,
    *,
    output_dir: Path = OUTPUT_DIR,
) -> ValidationSummary:
    """Run deterministic full-volume validation and save an epoch CSV."""
    validation_data = build_data_list(split="val")
    dataset = PersistentDataset(
        data=validation_data,
        transform=get_validation_transforms(),
        cache_dir=VALIDATION_CACHE_DIR,
        hash_transform=pickle_hashing,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    was_training = model.training
    model.eval()
    results: list[ValidationCaseResult] = []

    print("\n" + "=" * 72)
    print(
        f"EXPERIMENT V2 — EPOCH {epoch} "
        "RESEARCH-USE-ONLY VALIDATION"
    )
    print("=" * 72)
    print("Not for clinical diagnosis or treatment decisions.")
    print(f"Validation cases: {len(dataset)}")
    print(
        f"Sliding window: ROI={ROI_SIZE}, "
        f"sw_batch_size={SW_BATCH_SIZE}, overlap={OVERLAP}"
    )

    try:
        with torch.inference_mode():
            for case_number, batch in enumerate(loader, start=1):
                case_id = _single_string(batch["case_id"], "case ID")
                lesion_code = _single_string(
                    batch["lesion_code"],
                    "lesion code",
                )
                images = batch["image"]
                labels = batch["label"]
                _check_inputs(images, labels, case_id)

                # Keep full volumes and stitched output on CPU. Only ROI
                # batches are transferred to the GPU for inference.
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
                        device=torch.device("cpu"),
                    )

                if logits.shape != labels.shape:
                    raise ValueError(
                        f"Prediction/label shape mismatch for {case_id}: "
                        f"{tuple(logits.shape)} vs {tuple(labels.shape)}"
                    )
                if not bool(torch.isfinite(logits).all().item()):
                    raise ValueError(
                        f"Non-finite predictions for {case_id}"
                    )

                probabilities = torch.sigmoid(logits.float())
                prediction = (
                    probabilities >= PREDICTION_THRESHOLD
                )
                dice = _dice_score(prediction, labels)

                total_voxels = labels.numel()
                predicted_percent = (
                    100.0 * prediction.sum().item() / total_voxels
                )
                true_percent = (
                    100.0 * (labels > 0).sum().item() / total_voxels
                )

                if not all(
                    torch.isfinite(torch.tensor(value)).item()
                    for value in (dice, predicted_percent, true_percent)
                ):
                    raise ValueError(
                        f"Non-finite validation metric for {case_id}"
                    )

                results.append(
                    ValidationCaseResult(
                        case_id=case_id,
                        lesion_code=lesion_code,
                        dice=dice,
                        predicted_foreground_percent=predicted_percent,
                        true_foreground_percent=true_percent,
                    )
                )
                print(
                    f"[{case_number:03d}/{len(loader):03d}] "
                    f"{case_id} ({lesion_code}) | "
                    f"Dice={dice:.4f} | "
                    f"Pred FG={predicted_percent:.4f}% | "
                    f"True FG={true_percent:.4f}%"
                )
    finally:
        model.train(was_training)

    if not results:
        raise RuntimeError("Validation produced no case results.")

    mean_dice = statistics.fmean(
        result.dice for result in results
    )
    mean_predicted_percent = statistics.fmean(
        result.predicted_foreground_percent for result in results
    )
    mean_true_percent = statistics.fmean(
        result.true_foreground_percent for result in results
    )
    csv_path = _save_csv(results, epoch, output_dir)

    print("\nValidation summary")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(
        "Average predicted foreground: "
        f"{mean_predicted_percent:.6f}%"
    )
    print(
        f"Average true foreground: {mean_true_percent:.6f}%"
    )
    print(f"Validation CSV: {csv_path}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")

    return ValidationSummary(
        epoch=epoch,
        mean_dice=mean_dice,
        mean_predicted_foreground_percent=mean_predicted_percent,
        mean_true_foreground_percent=mean_true_percent,
        csv_path=csv_path,
        case_results=tuple(results),
    )


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    """Load one v2 experiment checkpoint for standalone validation."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Missing model_state_dict in {checkpoint_path}"
        )

    epoch = int(checkpoint.get("epoch", 0))
    if epoch < 1:
        raise ValueError(
            f"Invalid or missing epoch in {checkpoint_path}"
        )

    model = create_unet3d().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, epoch


def main() -> None:
    """Validate a saved v2 experiment checkpoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Research-use-only validation for OralVisionAI experiment v2."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="V2 checkpoint to evaluate.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, epoch = _load_checkpoint_model(args.checkpoint, device)

    print("OralVisionAI experiment v2 — RESEARCH USE ONLY")
    print(f"Loaded checkpoint: {args.checkpoint.resolve()}")
    validate_model(model, device, epoch)


if __name__ == "__main__":
    main()
