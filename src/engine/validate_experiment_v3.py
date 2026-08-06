"""Research-use-only full-volume validation for Experiment V3."""

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
THRESHOLD = 0.5
OUTPUT_DIR = Path("outputs/experiment_v3")
VALIDATION_CACHE_DIR = Path("data/cache/validation_v3")
DEFAULT_CHECKPOINT = Path("checkpoints/experiment_v3/best.pt")


@dataclass(frozen=True)
class ValidationResult:
    case_id: str
    lesion_code: str
    dice: float
    predicted_foreground_percent: float
    true_foreground_percent: float


@dataclass(frozen=True)
class ValidationSummary:
    epoch: int
    mean_dice: float
    mean_predicted_foreground_percent: float
    mean_true_foreground_percent: float
    csv_path: Path


def _metadata_string(value: object, name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0])
    raise ValueError(f"Expected one {name}, got {value!r}")


def _check_case(
    images: torch.Tensor,
    labels: torch.Tensor,
    case_id: str,
) -> None:
    if images.ndim != 5 or tuple(images.shape[:2]) != (1, 1):
        raise ValueError(
            f"Unexpected image shape for {case_id}: {tuple(images.shape)}"
        )
    if images.shape != labels.shape:
        raise ValueError(
            f"Image/label mismatch for {case_id}: "
            f"{tuple(images.shape)} vs {tuple(labels.shape)}"
        )
    if not bool(torch.isfinite(images).all().item()):
        raise ValueError(f"Non-finite image for {case_id}")
    if not bool(torch.isfinite(labels).all().item()):
        raise ValueError(f"Non-finite label for {case_id}")
    if not bool(torch.logical_or(labels == 0, labels == 1).all().item()):
        raise ValueError(f"Non-binary label for {case_id}")


def _dice(prediction: torch.Tensor, label: torch.Tensor) -> float:
    prediction = prediction.bool()
    label = label.bool()
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
    results: list[ValidationResult],
    epoch: int,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"validation_epoch_{epoch:03d}.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
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
    return path


def validate_model(
    model: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> ValidationSummary:
    """Validate one V3 epoch using full-volume sliding windows."""
    dataset = PersistentDataset(
        data=build_data_list(split="val"),
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
    previous_mode = model.training
    model.eval()
    results: list[ValidationResult] = []

    print("\n" + "=" * 72)
    print(f"EXPERIMENT V3 EPOCH {epoch} — RESEARCH-USE-ONLY VALIDATION")
    print("=" * 72)

    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader, start=1):
                case_id = _metadata_string(batch["case_id"], "case ID")
                lesion_code = _metadata_string(
                    batch["lesion_code"], "lesion code"
                )
                images = batch["image"]
                labels = batch["label"]
                _check_case(images, labels, case_id)

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
                        f"Prediction shape mismatch for {case_id}"
                    )
                if not bool(torch.isfinite(logits).all().item()):
                    raise ValueError(
                        f"Non-finite prediction for {case_id}"
                    )

                prediction = torch.sigmoid(logits.float()) >= THRESHOLD
                dice = _dice(prediction, labels)
                voxel_count = labels.numel()
                predicted_percent = (
                    100.0 * prediction.sum().item() / voxel_count
                )
                true_percent = (
                    100.0 * (labels > 0).sum().item() / voxel_count
                )
                results.append(
                    ValidationResult(
                        case_id,
                        lesion_code,
                        dice,
                        predicted_percent,
                        true_percent,
                    )
                )
                print(
                    f"[{index:03d}/{len(loader):03d}] {case_id} | "
                    f"Dice={dice:.4f} | Pred FG={predicted_percent:.4f}% | "
                    f"True FG={true_percent:.4f}%"
                )
    finally:
        model.train(previous_mode)

    if not results:
        raise RuntimeError("Validation produced no results.")

    mean_dice = statistics.fmean(item.dice for item in results)
    mean_predicted = statistics.fmean(
        item.predicted_foreground_percent for item in results
    )
    mean_true = statistics.fmean(
        item.true_foreground_percent for item in results
    )
    csv_path = _save_csv(results, epoch)

    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Average predicted foreground: {mean_predicted:.6f}%")
    print(f"Average true foreground: {mean_true:.6f}%")
    print(f"CSV saved to: {csv_path}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")
    return ValidationSummary(
        epoch,
        mean_dice,
        mean_predicted,
        mean_true,
        csv_path,
    )


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("experiment") != "v3_prevalence_mismatch":
        raise ValueError(f"Not a V3 checkpoint: {checkpoint_path}")
    epoch = int(checkpoint["epoch"])
    model = create_unet3d().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, epoch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research-use-only Experiment V3 validation."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    args = parser.parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, epoch = _load_model(args.checkpoint, device)
    print("OralVisionAI Experiment V3 — RESEARCH USE ONLY")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    validate_model(model, device, epoch)


if __name__ == "__main__":
    main()
