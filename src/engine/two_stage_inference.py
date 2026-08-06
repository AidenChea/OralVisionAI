"""Two-stage localization + segmentation inference for Experiment V4.

Stage 1 uses the coarse 128^3 localizer to estimate a lesion center.
Stage 2 runs the fine segmentation model (cached_epoch_002 by default) on a
128^3 crop around that center and writes the prediction back into native
full-volume coordinates.

Research use only — not for clinical diagnosis or treatment.

Run from the project root::

    python -m src.engine.two_stage_inference
    python -m src.engine.two_stage_inference --localizer checkpoints/localizer_v1/best.pt
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.transforms import ScaleIntensityRangePercentiles

from src.data.cached_dataset import build_data_list
from src.data.localization_transforms import (
    LOCALIZER_SPATIAL_SIZE,
    downsample_coords_to_original,
    get_localization_transforms,
)
from src.engine.validate_localizer import (
    PREDICTION_THRESHOLD,
    crop_overlaps_mask,
    dice_score,
    largest_connected_component,
    load_localizer_checkpoint,
    mask_centroid,
    millimeter_distance,
    predict_localizer_probabilities,
)
from src.models.unet3d import create_unet3d


OUTPUT_DIR = Path("outputs/two_stage_v1")
DEFAULT_LOCALIZER = Path("checkpoints/localizer_v1/best.pt")
DEFAULT_SEGMENTER = Path("checkpoints/cached_epoch_002.pt")
LESION_CLASSES: tuple[str, ...] = ("AME", "DC", "KCOT", "RC")
SEGMENT_CROP_SIZE = 128


@dataclass
class TwoStageCaseResult:
    """Full two-stage validation metrics for one case."""

    case_id: str
    lesion_code: str
    full_volume_dice: float
    localization_success: bool
    center_distance_mm: float
    predicted_foreground_percent: float
    true_foreground_percent: float
    true_lesion_voxels: int
    lesion_size_quartile: str
    predicted_center_original: tuple[float, float, float]
    true_center_original: tuple[float, float, float]


def _centered_crop_with_padding(
    volume: np.ndarray,
    center: tuple[float, float, float],
    size: int,
    pad_value: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return an exactly sized crop and the unpadded destination origin."""
    starts = [int(round(center[axis])) - size // 2 for axis in range(3)]
    ends = [start + size for start in starts]
    source_starts = [max(0, starts[axis]) for axis in range(3)]
    source_ends = [
        min(volume.shape[axis], ends[axis]) for axis in range(3)
    ]
    crop = volume[
        source_starts[0]:source_ends[0],
        source_starts[1]:source_ends[1],
        source_starts[2]:source_ends[2],
    ]
    padding = tuple(
        (
            max(0, -starts[axis]),
            max(0, ends[axis] - volume.shape[axis]),
        )
        for axis in range(3)
    )
    crop = np.pad(
        crop,
        pad_width=padding,
        mode="constant",
        constant_values=pad_value,
    )
    if crop.shape != (size, size, size):
        raise ValueError(
            f"Expected crop shape {(size, size, size)}, got {crop.shape}"
        )
    return np.asarray(crop, dtype=np.float32), (
        starts[0],
        starts[1],
        starts[2],
    )


def _insert_crop(
    full_volume: np.ndarray,
    crop: np.ndarray,
    origin: tuple[int, int, int],
) -> np.ndarray:
    """Write a crop back into a full-volume array with boundary clipping."""
    output = full_volume.copy()
    size = crop.shape
    for axis in range(3):
        if size[axis] <= 0:
            raise ValueError("Cannot insert an empty crop.")

    dest_starts = [max(0, origin[axis]) for axis in range(3)]
    dest_ends = [
        min(output.shape[axis], origin[axis] + size[axis])
        for axis in range(3)
    ]
    src_starts = [
        dest_starts[axis] - origin[axis] for axis in range(3)
    ]
    src_ends = [
        src_starts[axis] + (dest_ends[axis] - dest_starts[axis])
        for axis in range(3)
    ]
    if any(dest_ends[axis] <= dest_starts[axis] for axis in range(3)):
        return output

    output[
        dest_starts[0]:dest_ends[0],
        dest_starts[1]:dest_ends[1],
        dest_starts[2]:dest_ends[2],
    ] = crop[
        src_starts[0]:src_ends[0],
        src_starts[1]:src_ends[1],
        src_starts[2]:src_ends[2],
    ]
    return output


def _normalize_crop(image_crop: np.ndarray) -> torch.Tensor:
    """Apply training-equivalent percentile normalization to one crop."""
    normalizer = ScaleIntensityRangePercentiles(
        lower=1,
        upper=99,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    normalized = normalizer(image_crop[np.newaxis, ...])
    tensor = torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0)
    expected = (1, 1, SEGMENT_CROP_SIZE, SEGMENT_CROP_SIZE, SEGMENT_CROP_SIZE)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"Unexpected normalized crop shape: {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("Normalized crop contains non-finite values.")
    return tensor


def _load_native_case(
    case: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load native-resolution image, binary label, and spacing."""
    image_nifti = nib.load(case["image"])
    label_nifti = nib.load(case["label"])
    image = image_nifti.get_fdata(dtype=np.float32)
    label = (label_nifti.get_fdata(dtype=np.float32) > 0).astype(np.float32)
    spacing = tuple(
        float(value) for value in image_nifti.header.get_zooms()[:3]
    )

    if image.shape != label.shape:
        raise ValueError(
            f"Shape mismatch for {case['case_id']}: "
            f"{image.shape} vs {label.shape}"
        )
    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D volume for {case['case_id']}, got {image.shape}"
        )
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite image for {case['case_id']}")
    if not np.isfinite(label).all():
        raise ValueError(f"Non-finite label for {case['case_id']}")
    if int(label.sum()) == 0:
        raise ValueError(f"Empty label for {case['case_id']}")
    return image, label, spacing


def _load_segmenter(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Missing model_state_dict in {checkpoint_path}")
    model = create_unet3d().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _assign_quartiles(
    results: list[TwoStageCaseResult],
) -> tuple[float, ...]:
    sizes = [result.true_lesion_voxels for result in results]
    boundaries = tuple(
        float(value) for value in np.quantile(sizes, [0.25, 0.5, 0.75])
    )
    for result in results:
        quartile = (
            int(np.searchsorted(boundaries, result.true_lesion_voxels, side="right"))
            + 1
        )
        result.lesion_size_quartile = f"Q{quartile}"
    return boundaries


def _summarize(
    name: str,
    subset: list[TwoStageCaseResult],
) -> None:
    if not subset:
        print(f"{name}: no cases")
        return
    distances = [
        result.center_distance_mm
        for result in subset
        if np.isfinite(result.center_distance_mm)
    ]
    print(
        f"{name}: n={len(subset)} | "
        f"Dice={statistics.fmean(result.full_volume_dice for result in subset):.4f} | "
        f"LocSuccess={100.0 * statistics.fmean(float(result.localization_success) for result in subset):.1f}% | "
        f"Dist={statistics.fmean(distances) if distances else float('nan'):.2f} mm | "
        f"PredFG={statistics.fmean(result.predicted_foreground_percent for result in subset):.4f}% | "
        f"TrueFG={statistics.fmean(result.true_foreground_percent for result in subset):.4f}%"
    )


def _save_results(results: list[TwoStageCaseResult]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "two_stage_validation.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "lesion_code",
                "full_volume_dice",
                "localization_success",
                "center_distance_mm",
                "predicted_foreground_percent",
                "true_foreground_percent",
                "true_lesion_voxels",
                "lesion_size_quartile",
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
                    "full_volume_dice": f"{result.full_volume_dice:.8f}",
                    "localization_success": str(
                        result.localization_success
                    ).lower(),
                    "center_distance_mm": (
                        f"{result.center_distance_mm:.8f}"
                        if np.isfinite(result.center_distance_mm)
                        else ""
                    ),
                    "predicted_foreground_percent": (
                        f"{result.predicted_foreground_percent:.8f}"
                    ),
                    "true_foreground_percent": (
                        f"{result.true_foreground_percent:.8f}"
                    ),
                    "true_lesion_voxels": result.true_lesion_voxels,
                    "lesion_size_quartile": result.lesion_size_quartile,
                    "predicted_center_x": (
                        f"{result.predicted_center_original[0]:.4f}"
                    ),
                    "predicted_center_y": (
                        f"{result.predicted_center_original[1]:.4f}"
                    ),
                    "predicted_center_z": (
                        f"{result.predicted_center_original[2]:.4f}"
                    ),
                    "true_center_x": (
                        f"{result.true_center_original[0]:.4f}"
                    ),
                    "true_center_y": (
                        f"{result.true_center_original[1]:.4f}"
                    ),
                    "true_center_z": (
                        f"{result.true_center_original[2]:.4f}"
                    ),
                    "research_use_only": "true",
                }
            )
    return path


def run_two_stage_case(
    *,
    case: dict[str, str],
    localizer: torch.nn.Module,
    segmenter: torch.nn.Module,
    device: torch.device,
    localization_transform,
) -> TwoStageCaseResult:
    """Run localization + cropped segmentation for one validation case."""
    case_id = case["case_id"]
    lesion_code = case["lesion_code"]
    if lesion_code not in LESION_CLASSES:
        raise ValueError(f"Unexpected lesion class for {case_id}")

    native_image, native_label, spacing_mm = _load_native_case(case)
    original_shape = tuple(int(size) for size in native_image.shape)
    true_center_original = mask_centroid(native_label)
    if true_center_original is None:
        raise ValueError(f"Empty native label for {case_id}")

    # Stage 1: coarse localization on the downsampled volume.
    localized = localization_transform(case)
    image_small = localized["image"]
    if not isinstance(image_small, torch.Tensor):
        raise TypeError(
            f"Expected tensor localizer input for {case_id}"
        )
    if tuple(image_small.shape[-3:]) != LOCALIZER_SPATIAL_SIZE:
        raise ValueError(
            f"Unexpected localizer input shape for {case_id}: "
            f"{tuple(image_small.shape)}"
        )

    probabilities = predict_localizer_probabilities(
        localizer,
        image_small.unsqueeze(0),
        device,
    )
    prediction_small = (
        probabilities.squeeze().numpy() >= PREDICTION_THRESHOLD
    ).astype(np.float32)
    prediction_small = largest_connected_component(prediction_small)
    predicted_center_small = mask_centroid(prediction_small)

    if predicted_center_small is None:
        predicted_center_original = tuple(
            (size - 1) / 2.0 for size in original_shape
        )
        localization_success = False
        center_distance = float("nan")
    else:
        predicted_center_original = downsample_coords_to_original(
            predicted_center_small,
            original_shape,
        )
        localization_success = crop_overlaps_mask(
            predicted_center_original,
            native_label,
            crop_size=SEGMENT_CROP_SIZE,
        )
        center_distance = millimeter_distance(
            predicted_center_original,
            true_center_original,
            spacing_mm,
        )

    # Stage 2: fine segmentation inside the predicted-center crop.
    image_crop, origin = _centered_crop_with_padding(
        native_image,
        predicted_center_original,
        SEGMENT_CROP_SIZE,
        pad_value=float(native_image.min()),
    )
    crop_tensor = _normalize_crop(image_crop)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = segmenter(crop_tensor.to(device))
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError(f"Non-finite segmenter logits for {case_id}")

    crop_probability = torch.sigmoid(logits.float()).cpu().squeeze().numpy()
    crop_prediction = (
        crop_probability >= PREDICTION_THRESHOLD
    ).astype(np.float32)

    full_prediction = np.zeros(original_shape, dtype=np.float32)
    full_prediction = _insert_crop(
        full_prediction,
        crop_prediction,
        origin,
    )

    full_dice = dice_score(full_prediction, native_label)
    voxel_count = native_label.size
    predicted_percent = 100.0 * float(full_prediction.sum()) / voxel_count
    true_percent = 100.0 * float(native_label.sum()) / voxel_count

    return TwoStageCaseResult(
        case_id=case_id,
        lesion_code=lesion_code,
        full_volume_dice=full_dice,
        localization_success=localization_success,
        center_distance_mm=center_distance,
        predicted_foreground_percent=predicted_percent,
        true_foreground_percent=true_percent,
        true_lesion_voxels=int(native_label.sum()),
        lesion_size_quartile="",
        predicted_center_original=predicted_center_original,
        true_center_original=true_center_original,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Research-use-only two-stage OralVisionAI inference."
        )
    )
    parser.add_argument(
        "--localizer",
        type=Path,
        default=DEFAULT_LOCALIZER,
        help="Localizer checkpoint path.",
    )
    parser.add_argument(
        "--segmenter",
        type=Path,
        default=DEFAULT_SEGMENTER,
        help="Fine segmentation checkpoint path.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    localizer, localizer_epoch = load_localizer_checkpoint(
        args.localizer,
        device,
    )
    segmenter = _load_segmenter(args.segmenter, device)
    localization_transform = get_localization_transforms()
    validation_cases = build_data_list(split="val")

    print("=" * 72)
    print("OralVision AI — TWO-STAGE V1 INFERENCE")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Localizer: {args.localizer.resolve()} (epoch {localizer_epoch})")
    print(f"Segmenter: {args.segmenter.resolve()}")
    print(f"Validation cases: {len(validation_cases)}")
    print(
        f"Localization grid: {LOCALIZER_SPATIAL_SIZE} | "
        f"Segmentation crop: {SEGMENT_CROP_SIZE}^3"
    )

    results: list[TwoStageCaseResult] = []
    for index, case in enumerate(validation_cases, start=1):
        result = run_two_stage_case(
            case=case,
            localizer=localizer,
            segmenter=segmenter,
            device=device,
            localization_transform=localization_transform,
        )
        results.append(result)
        distance_text = (
            f"{result.center_distance_mm:.2f} mm"
            if np.isfinite(result.center_distance_mm)
            else "n/a"
        )
        print(
            f"[{index:03d}/{len(validation_cases):03d}] "
            f"{result.case_id} ({result.lesion_code}) | "
            f"Dice={result.full_volume_dice:.4f} | "
            f"LocSuccess={result.localization_success} | "
            f"Dist={distance_text} | "
            f"PredFG={result.predicted_foreground_percent:.4f}%"
        )

    boundaries = _assign_quartiles(results)
    csv_path = _save_results(results)

    print("\n" + "=" * 72)
    print("TWO-STAGE VALIDATION SUMMARY — RESEARCH USE ONLY")
    print("=" * 72)
    _summarize("Overall", results)
    for lesion_code in LESION_CLASSES:
        subset = [
            result
            for result in results
            if result.lesion_code == lesion_code
        ]
        _summarize(lesion_code, subset)
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        subset = [
            result
            for result in results
            if result.lesion_size_quartile == quartile
        ]
        _summarize(quartile, subset)

    print(f"\nLesion-size quartile boundaries (voxels): {boundaries}")
    print(f"Results CSV: {csv_path.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
