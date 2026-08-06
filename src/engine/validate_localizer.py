"""Validate the OralVisionAI coarse lesion localizer.

Reports coarse Dice, predicted-center distance in millimeters, and case-level
localization success rates for research-use-only Experiment V4.

Run from the project root::

    python -m src.engine.validate_localizer
    python -m src.engine.validate_localizer --checkpoint checkpoints/localizer_v1/best.pt
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.data.utils import pickle_hashing
from scipy import ndimage

from src.data.cached_dataset import build_data_list
from src.data.localization_transforms import (
    LOCALIZER_CACHE_DIR,
    LOCALIZER_SPATIAL_SIZE,
    downsample_coords_to_original,
    get_localization_transforms,
)
from src.models.localizer3d import create_localizer3d


CHECKPOINT_DIR = Path("checkpoints/localizer_v1")
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
OUTPUT_DIR = Path("outputs/two_stage_v1")
PREDICTION_THRESHOLD = 0.5
CROP_SIZE_ORIGINAL = 128
LESION_CLASSES: tuple[str, ...] = ("AME", "DC", "KCOT", "RC")


@dataclass(frozen=True)
class LocalizationCaseResult:
    """Per-case coarse localization metrics."""

    case_id: str
    lesion_code: str
    coarse_dice: float
    center_distance_mm: float
    predicted_center_inside_lesion: bool
    crop_128_overlaps_lesion: bool
    lesion_detected: bool
    true_lesion_voxels_original: int
    predicted_center_original: tuple[float, float, float]
    true_center_original: tuple[float, float, float]


@dataclass(frozen=True)
class LocalizationSummary:
    """Aggregate localization metrics for one validation pass."""

    epoch: int
    mean_coarse_dice: float
    mean_center_distance_mm: float
    center_inside_lesion_rate: float
    crop_overlap_rate: float
    lesion_recall: float
    csv_path: Path
    case_results: tuple[LocalizationCaseResult, ...]


def _single_string(value: object, name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0])
    raise ValueError(f"Expected one {name}, got {value!r}")


def _unwrap_batch_value(value: object) -> object:
    """Unwrap length-1 collated metadata containers."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _as_shape_tuple(value: object) -> tuple[int, int, int]:
    value = _unwrap_batch_value(value)
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().flatten().tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise TypeError(f"Unexpected geometry value: {type(value)}")
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {values!r}")
    return int(values[0]), int(values[1]), int(values[2])


def _as_spacing_tuple(value: object) -> tuple[float, float, float]:
    value = _unwrap_batch_value(value)
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().flatten().tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise TypeError(f"Unexpected spacing value: {type(value)}")
    if len(values) != 3:
        raise ValueError(f"Expected 3 spacing values, got {values!r}")
    return float(values[0]), float(values[1]), float(values[2])


def _check_localizer_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    case_id: str,
) -> None:
    expected = (1, 1, *LOCALIZER_SPATIAL_SIZE)
    if tuple(images.shape) != expected:
        raise ValueError(
            f"Unexpected localizer image shape for {case_id}: "
            f"{tuple(images.shape)}"
        )
    if tuple(labels.shape) != expected:
        raise ValueError(
            f"Unexpected localizer label shape for {case_id}: "
            f"{tuple(labels.shape)}"
        )
    if not bool(torch.isfinite(images).all().item()):
        raise ValueError(f"Non-finite localizer image for {case_id}")
    if not bool(torch.isfinite(labels).all().item()):
        raise ValueError(f"Non-finite localizer label for {case_id}")
    if not bool(torch.logical_or(labels == 0, labels == 1).all().item()):
        raise ValueError(f"Non-binary localizer label for {case_id}")


def dice_score(
    prediction: torch.Tensor | np.ndarray,
    label: torch.Tensor | np.ndarray,
) -> float:
    """Binary Dice; empty-empty is treated as a perfect match."""
    if isinstance(prediction, torch.Tensor):
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

    prediction_bool = prediction.astype(bool)
    label_bool = label.astype(bool)
    intersection = float(np.logical_and(prediction_bool, label_bool).sum())
    denominator = float(prediction_bool.sum() + label_bool.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator


def mask_centroid(
    mask: np.ndarray,
) -> tuple[float, float, float] | None:
    """Return the center of mass for a binary mask, or None if empty."""
    if mask.sum() <= 0:
        return None
    center = ndimage.center_of_mass(mask.astype(np.float64))
    return float(center[0]), float(center[1]), float(center[2])


def largest_connected_component(
    mask: np.ndarray,
) -> np.ndarray:
    """Keep only the largest 26-connected foreground component."""
    binary = mask.astype(bool)
    if not binary.any():
        return binary.astype(np.float32)

    labeled, count = ndimage.label(binary)
    if count <= 1:
        return binary.astype(np.float32)

    sizes = ndimage.sum(binary, labeled, index=range(1, count + 1))
    largest_index = int(np.argmax(sizes)) + 1
    return (labeled == largest_index).astype(np.float32)


def millimeter_distance(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    spacing_mm: tuple[float, float, float],
) -> float:
    """Euclidean distance between two voxel points, converted to millimeters."""
    squared = 0.0
    for index in range(3):
        delta_mm = (point_a[index] - point_b[index]) * spacing_mm[index]
        squared += delta_mm * delta_mm
    return float(np.sqrt(squared))


def point_inside_mask(
    point: tuple[float, float, float],
    mask: np.ndarray,
) -> bool:
    """True when the nearest voxel to *point* is foreground."""
    if mask.sum() <= 0:
        return False
    rounded = [
        int(np.clip(round(point[index]), 0, mask.shape[index] - 1))
        for index in range(3)
    ]
    return bool(mask[rounded[0], rounded[1], rounded[2]] > 0)


def crop_overlaps_mask(
    center: tuple[float, float, float],
    mask: np.ndarray,
    crop_size: int = CROP_SIZE_ORIGINAL,
) -> bool:
    """True when a cubic crop around *center* intersects the mask."""
    half = crop_size // 2
    slices: list[slice] = []
    for index in range(3):
        start = int(round(center[index])) - half
        end = start + crop_size
        start = max(0, start)
        end = min(mask.shape[index], end)
        if end <= start:
            return False
        slices.append(slice(start, end))
    return bool(mask[tuple(slices)].sum() > 0)


def estimate_original_lesion_voxels(
    downsampled_label: np.ndarray,
    original_shape: tuple[int, int, int],
) -> int:
    """Approximate native lesion voxel count from the downsampled mask."""
    scale = float(np.prod(original_shape) / np.prod(LOCALIZER_SPATIAL_SIZE))
    return int(round(float(downsampled_label.sum()) * scale))


def predict_localizer_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run localizer inference and return sigmoid probabilities on CPU."""
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images.to(device, non_blocking=True))
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("Localizer produced non-finite logits.")
    probabilities = torch.sigmoid(logits.float()).cpu()
    if not bool(torch.isfinite(probabilities).all().item()):
        raise ValueError("Localizer produced non-finite probabilities.")
    return probabilities


def localization_metrics_from_probability(
    *,
    probability: torch.Tensor,
    label: torch.Tensor,
    original_shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    case_id: str,
    lesion_code: str,
    threshold: float = PREDICTION_THRESHOLD,
) -> LocalizationCaseResult:
    """Compute all case-level localization metrics from one prediction."""
    probability_volume = probability.squeeze().detach().cpu().numpy()
    label_volume = label.squeeze().detach().cpu().numpy()
    if probability_volume.shape != LOCALIZER_SPATIAL_SIZE:
        raise ValueError(
            f"Unexpected probability shape for {case_id}: "
            f"{probability_volume.shape}"
        )
    if label_volume.shape != LOCALIZER_SPATIAL_SIZE:
        raise ValueError(
            f"Unexpected label shape for {case_id}: {label_volume.shape}"
        )

    prediction = (probability_volume >= threshold).astype(np.float32)
    prediction = largest_connected_component(prediction)
    coarse_dice = dice_score(prediction, label_volume)

    true_center_down = mask_centroid(label_volume)
    if true_center_down is None:
        raise ValueError(f"Empty localizer label for {case_id}")

    predicted_center_down = mask_centroid(prediction)
    lesion_detected = predicted_center_down is not None

    true_center_original = downsample_coords_to_original(
        true_center_down,
        original_shape,
    )

    if predicted_center_down is None:
        # Fall back to volume center so distance remains finite and comparable.
        predicted_center_down = tuple(
            (size - 1) / 2.0 for size in LOCALIZER_SPATIAL_SIZE
        )
        predicted_center_original = downsample_coords_to_original(
            predicted_center_down,
            original_shape,
        )
        center_inside = False
        crop_overlaps = False
        center_distance = float("nan")
    else:
        predicted_center_original = downsample_coords_to_original(
            predicted_center_down,
            original_shape,
        )
        # Build an approximate original-resolution mask by nearest upsampling
        # for center-in-lesion and crop-overlap checks.
        zoom_factors = tuple(
            original_shape[index] / LOCALIZER_SPATIAL_SIZE[index]
            for index in range(3)
        )
        original_label = ndimage.zoom(
            label_volume,
            zoom=zoom_factors,
            order=0,
        )
        if original_label.shape != original_shape:
            # Guard against rounding drift from zoom.
            resized = np.zeros(original_shape, dtype=np.float32)
            slices = tuple(
                slice(0, min(original_shape[index], original_label.shape[index]))
                for index in range(3)
            )
            resized[slices] = original_label[slices]
            original_label = resized

        center_inside = point_inside_mask(
            predicted_center_original,
            original_label,
        )
        crop_overlaps = crop_overlaps_mask(
            predicted_center_original,
            original_label,
            crop_size=CROP_SIZE_ORIGINAL,
        )
        center_distance = millimeter_distance(
            predicted_center_original,
            true_center_original,
            spacing_mm,
        )

    return LocalizationCaseResult(
        case_id=case_id,
        lesion_code=lesion_code,
        coarse_dice=coarse_dice,
        center_distance_mm=center_distance,
        predicted_center_inside_lesion=center_inside,
        crop_128_overlaps_lesion=crop_overlaps,
        lesion_detected=lesion_detected,
        true_lesion_voxels_original=estimate_original_lesion_voxels(
            label_volume,
            original_shape,
        ),
        predicted_center_original=predicted_center_original,
        true_center_original=true_center_original,
    )


def _save_results(
    results: Sequence[LocalizationCaseResult],
    epoch: int,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"localizer_validation_epoch_{epoch:03d}.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "lesion_code",
                "coarse_dice",
                "center_distance_mm",
                "predicted_center_inside_lesion",
                "crop_128_overlaps_lesion",
                "lesion_detected",
                "true_lesion_voxels_original",
                "predicted_center_x",
                "predicted_center_y",
                "predicted_center_z",
                "true_center_x",
                "true_center_y",
                "true_center_z",
                "research_use_only",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "lesion_code": result.lesion_code,
                    "coarse_dice": f"{result.coarse_dice:.8f}",
                    "center_distance_mm": (
                        f"{result.center_distance_mm:.8f}"
                        if np.isfinite(result.center_distance_mm)
                        else ""
                    ),
                    "predicted_center_inside_lesion": str(
                        result.predicted_center_inside_lesion
                    ).lower(),
                    "crop_128_overlaps_lesion": str(
                        result.crop_128_overlaps_lesion
                    ).lower(),
                    "lesion_detected": str(result.lesion_detected).lower(),
                    "true_lesion_voxels_original": (
                        result.true_lesion_voxels_original
                    ),
                    "predicted_center_x": (
                        f"{result.predicted_center_original[0]:.4f}"
                    ),
                    "predicted_center_y": (
                        f"{result.predicted_center_original[1]:.4f}"
                    ),
                    "predicted_center_z": (
                        f"{result.predicted_center_original[2]:.4f}"
                    ),
                    "true_center_x": f"{result.true_center_original[0]:.4f}",
                    "true_center_y": f"{result.true_center_original[1]:.4f}",
                    "true_center_z": f"{result.true_center_original[2]:.4f}",
                    "research_use_only": "true",
                }
            )
    return path


def validate_localizer(
    model: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> LocalizationSummary:
    """Run coarse localization validation on the val split."""
    dataset = PersistentDataset(
        data=build_data_list(split="val"),
        transform=get_localization_transforms(),
        cache_dir=LOCALIZER_CACHE_DIR,
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
    results: list[LocalizationCaseResult] = []

    print("\n" + "=" * 72)
    print(
        f"LOCALIZER V1 EPOCH {epoch} — "
        "RESEARCH-USE-ONLY VALIDATION"
    )
    print("=" * 72)
    print("Not for clinical diagnosis or treatment decisions.")
    print(f"Validation cases: {len(dataset)}")
    print(f"Downsampled grid: {LOCALIZER_SPATIAL_SIZE}")

    try:
        for index, batch in enumerate(loader, start=1):
            case_id = _single_string(batch["case_id"], "case ID")
            lesion_code = _single_string(
                batch["lesion_code"],
                "lesion code",
            )
            images = batch["image"]
            labels = batch["label"]
            _check_localizer_batch(images, labels, case_id)

            original_shape = _as_shape_tuple(batch["original_shape"])
            spacing_mm = _as_spacing_tuple(batch["spacing_mm"])

            probabilities = predict_localizer_probabilities(
                model,
                images,
                device,
            )
            result = localization_metrics_from_probability(
                probability=probabilities[0],
                label=labels[0],
                original_shape=original_shape,
                spacing_mm=spacing_mm,
                case_id=case_id,
                lesion_code=lesion_code,
            )
            results.append(result)
            distance_text = (
                f"{result.center_distance_mm:.2f} mm"
                if np.isfinite(result.center_distance_mm)
                else "n/a"
            )
            print(
                f"[{index:03d}/{len(loader):03d}] {case_id} | "
                f"Dice={result.coarse_dice:.4f} | "
                f"Dist={distance_text} | "
                f"Inside={result.predicted_center_inside_lesion} | "
                f"CropHit={result.crop_128_overlaps_lesion}"
            )
    finally:
        model.train(previous_mode)

    if not results:
        raise RuntimeError("Localizer validation produced no results.")

    finite_distances = [
        result.center_distance_mm
        for result in results
        if np.isfinite(result.center_distance_mm)
    ]
    summary = LocalizationSummary(
        epoch=epoch,
        mean_coarse_dice=statistics.fmean(
            result.coarse_dice for result in results
        ),
        mean_center_distance_mm=(
            statistics.fmean(finite_distances)
            if finite_distances
            else float("nan")
        ),
        center_inside_lesion_rate=statistics.fmean(
            float(result.predicted_center_inside_lesion)
            for result in results
        ),
        crop_overlap_rate=statistics.fmean(
            float(result.crop_128_overlaps_lesion)
            for result in results
        ),
        lesion_recall=statistics.fmean(
            float(result.lesion_detected) for result in results
        ),
        csv_path=_save_results(results, epoch),
        case_results=tuple(results),
    )

    print("\nLocalization summary")
    print(f"Mean coarse Dice: {summary.mean_coarse_dice:.4f}")
    print(
        f"Mean center distance: "
        f"{summary.mean_center_distance_mm:.2f} mm"
    )
    print(
        "Predicted center inside lesion: "
        f"{100.0 * summary.center_inside_lesion_rate:.1f}%"
    )
    print(
        "128^3 crop overlaps lesion: "
        f"{100.0 * summary.crop_overlap_rate:.1f}%"
    )
    print(
        f"Case-level lesion recall: "
        f"{100.0 * summary.lesion_recall:.1f}%"
    )
    print(f"CSV saved to: {summary.csv_path}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")
    return summary


def load_localizer_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    """Load a trained localizer checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Missing model_state_dict in {checkpoint_path}")

    model = create_localizer3d().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    epoch = int(checkpoint.get("epoch", 0))
    return model, epoch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Research-use-only validation for the OralVisionAI localizer."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Localizer checkpoint to evaluate.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, epoch = load_localizer_checkpoint(args.checkpoint, device)

    print("OralVisionAI Localizer V1 — RESEARCH USE ONLY")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    validate_localizer(model, device, max(epoch, 1))


if __name__ == "__main__":
    main()
