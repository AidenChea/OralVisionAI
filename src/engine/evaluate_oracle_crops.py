"""Research-only expert-label oracle-crop evaluation.

This diagnostic uses the expert label to center each crop. Its results are
optimistic localization upper bounds and are not valid estimates of clinical
or autonomous full-volume performance.

Run from the project root::

    python -m src.engine.evaluate_oracle_crops
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import nibabel as nib
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensityRangePercentiles

from src.data.cached_dataset import build_data_list
from src.models.unet3d import create_unet3d


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


@dataclass
class MetricRecord:
    checkpoint: str
    checkpoint_path: str
    epoch: int
    case_id: str
    lesion_class: str
    full_lesion_voxels: int
    lesion_size_quartile: str
    crop_size: int
    threshold: float
    dice: float
    soft_dice: float
    precision: float
    recall: float
    predicted_foreground_percent: float
    true_foreground_percent: float
    research_use_only: bool = True


@dataclass(frozen=True)
class SummaryRecord:
    group_type: str
    group_value: str
    best_threshold: float
    evaluation_count: int
    unique_case_count: int
    mean_thresholded_dice: float
    mean_soft_dice: float
    mean_precision: float
    mean_recall: float
    mean_predicted_foreground_percent: float
    mean_true_foreground_percent: float
    research_use_only: bool = True


@dataclass(frozen=True)
class VisualSlice:
    image: np.ndarray
    label: np.ndarray
    probability: np.ndarray


CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        "cached_epoch_002",
        Path("checkpoints/cached_epoch_002.pt"),
    ),
    CheckpointSpec(
        "experiment_v3_epoch_001",
        Path("checkpoints/experiment_v3/epoch_001.pt"),
    ),
    CheckpointSpec(
        "experiment_v3_epoch_005",
        Path("checkpoints/experiment_v3/epoch_005.pt"),
    ),
)
CROP_SIZES: tuple[int, ...] = (96, 128, 192)
THRESHOLDS: tuple[float, ...] = tuple(
    round(value / 10, 1) for value in range(1, 10)
)
LESION_CLASSES: tuple[str, ...] = ("AME", "DC", "KCOT", "RC")
SLIDING_ROI: tuple[int, int, int] = (96, 96, 96)
SLIDING_BATCH_SIZE = 2
SLIDING_OVERLAP = 0.5
DETAIL_CSV = Path("outputs/oracle_crop_evaluation.csv")
SUMMARY_CSV = Path("outputs/oracle_crop_summary.csv")
FIGURE_DIR = Path("outputs/oracle_crop_figures")
FIGURE_CASE_COUNT = 5
EPSILON = 1e-8
PREDICTION_CMAP = ListedColormap(["cyan"])


def _load_checkpoint_models(
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, int]]:
    """Load all requested checkpoints and validate their model tensors."""
    models: dict[str, torch.nn.Module] = {}
    epochs: dict[str, int] = {}

    for spec in CHECKPOINTS:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        checkpoint: dict[str, Any] = torch.load(
            spec.path,
            map_location="cpu",
            weights_only=True,
        )
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"Missing model_state_dict in {spec.path}"
            )
        state = checkpoint["model_state_dict"]
        if not all(
            bool(torch.isfinite(tensor).all().item())
            for tensor in state.values()
        ):
            raise ValueError(
                f"Non-finite model parameters in {spec.path}"
            )

        model = create_unet3d()
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models[spec.name] = model
        epochs[spec.name] = int(checkpoint.get("epoch", 0))

    return models, epochs


def _load_case(
    case: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load full NIfTI image and binary expert label."""
    image = nib.load(case["image"]).get_fdata(dtype=np.float32)
    raw_label = nib.load(case["label"]).get_fdata(dtype=np.float32)

    if image.shape != raw_label.shape:
        raise ValueError(
            f"Full-volume shape mismatch for {case['case_id']}: "
            f"{image.shape} vs {raw_label.shape}"
        )
    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D volume for {case['case_id']}, "
            f"got {image.shape}"
        )
    if not np.isfinite(image).all():
        raise ValueError(
            f"Non-finite image values for {case['case_id']}"
        )
    if not np.isfinite(raw_label).all():
        raise ValueError(
            f"Non-finite label values for {case['case_id']}"
        )

    label = (raw_label > 0).astype(np.float32)
    if not np.isin(np.unique(label), [0.0, 1.0]).all():
        raise ValueError(
            f"Label binarization failed for {case['case_id']}"
        )
    if int(label.sum()) == 0:
        raise ValueError(
            f"Validation label is empty for {case['case_id']}"
        )
    return image, label


def _label_bbox_center(
    label: np.ndarray,
) -> tuple[tuple[int, int, int], tuple[int, ...], tuple[int, ...]]:
    """Return integer center and inclusive expert-label bounding box."""
    minimums: list[int] = []
    maximums: list[int] = []

    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        occupied = np.flatnonzero(np.any(label > 0, axis=other_axes))
        if occupied.size == 0:
            raise ValueError("Cannot compute bounding box of empty label.")
        minimums.append(int(occupied[0]))
        maximums.append(int(occupied[-1]))

    center = tuple(
        (minimum + maximum) // 2
        for minimum, maximum in zip(minimums, maximums)
    )
    return center, tuple(minimums), tuple(maximums)


def _centered_crop_with_padding(
    volume: np.ndarray,
    center: tuple[int, int, int],
    size: int,
    pad_value: float,
) -> np.ndarray:
    """Extract an exactly sized center crop, padding beyond boundaries."""
    starts = [center[axis] - size // 2 for axis in range(3)]
    ends = [start + size for start in starts]
    source_starts = [
        max(0, starts[axis]) for axis in range(3)
    ]
    source_ends = [
        min(volume.shape[axis], ends[axis]) for axis in range(3)
    ]
    slices = tuple(
        slice(source_starts[axis], source_ends[axis])
        for axis in range(3)
    )
    crop = volume[slices]
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
            f"Padded crop has shape {crop.shape}, "
            f"expected {(size, size, size)}"
        )
    return np.asarray(crop, dtype=np.float32)


def _prepare_crop(
    image: np.ndarray,
    label: np.ndarray,
    center: tuple[int, int, int],
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop, pad, and apply training-equivalent percentile scaling."""
    image_crop = _centered_crop_with_padding(
        image,
        center,
        size,
        pad_value=float(image.min()),
    )
    label_crop = _centered_crop_with_padding(
        label,
        center,
        size,
        pad_value=0.0,
    )
    label_crop = (label_crop > 0).astype(np.float32)

    normalizer = ScaleIntensityRangePercentiles(
        lower=1,
        upper=99,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    normalized = normalizer(image_crop[np.newaxis, ...])
    image_tensor = torch.as_tensor(
        normalized,
        dtype=torch.float32,
    ).unsqueeze(0)
    label_tensor = torch.from_numpy(
        label_crop[np.newaxis, np.newaxis, ...]
    )

    expected_shape = (1, 1, size, size, size)
    if tuple(image_tensor.shape) != expected_shape:
        raise ValueError(
            f"Normalized image shape {tuple(image_tensor.shape)} "
            f"does not match {expected_shape}"
        )
    if tuple(label_tensor.shape) != expected_shape:
        raise ValueError(
            f"Label crop shape {tuple(label_tensor.shape)} "
            f"does not match {expected_shape}"
        )
    if not bool(torch.isfinite(image_tensor).all().item()):
        raise ValueError("Normalized crop contains non-finite values.")
    if not bool(torch.isfinite(label_tensor).all().item()):
        raise ValueError("Label crop contains non-finite values.")
    return image_tensor, label_tensor


def _infer_probabilities(
    model: torch.nn.Module,
    image: torch.Tensor,
    crop_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Use direct inference except for memory-sensitive 192-voxel crops."""
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            if crop_size == 192:
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=SLIDING_ROI,
                    sw_batch_size=SLIDING_BATCH_SIZE,
                    predictor=model,
                    overlap=SLIDING_OVERLAP,
                    sw_device=device,
                    device=torch.device("cpu"),
                )
            else:
                try:
                    logits = model(image.to(device)).cpu()
                except torch.OutOfMemoryError:
                    if device.type != "cuda":
                        raise
                    torch.cuda.empty_cache()
                    print(
                        f"Direct {crop_size}³ inference exceeded VRAM; "
                        "using sliding-window fallback."
                    )
                    logits = sliding_window_inference(
                        inputs=image,
                        roi_size=SLIDING_ROI,
                        sw_batch_size=SLIDING_BATCH_SIZE,
                        predictor=model,
                        overlap=SLIDING_OVERLAP,
                        sw_device=device,
                        device=torch.device("cpu"),
                    )

    if logits.shape != image.shape:
        raise ValueError(
            f"Logit/image mismatch: "
            f"{tuple(logits.shape)} vs {tuple(image.shape)}"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("Inference produced non-finite logits.")
    probabilities = torch.sigmoid(logits.float())
    if not bool(torch.isfinite(probabilities).all().item()):
        raise ValueError("Inference produced non-finite probabilities.")
    return probabilities


def _soft_dice(
    probabilities: torch.Tensor,
    label: torch.Tensor,
) -> float:
    intersection = (probabilities * label).sum(dtype=torch.float64)
    denominator = (
        probabilities.sum(dtype=torch.float64)
        + label.sum(dtype=torch.float64)
    )
    return float(
        ((2.0 * intersection + EPSILON) / (denominator + EPSILON)).item()
    )


def _threshold_metrics(
    probabilities: torch.Tensor,
    label: torch.Tensor,
    threshold: float,
) -> tuple[float, float, float, float, float]:
    prediction = probabilities >= threshold
    truth = label > 0
    true_positive = torch.logical_and(prediction, truth).sum().item()
    false_positive = torch.logical_and(
        prediction,
        torch.logical_not(truth),
    ).sum().item()
    false_negative = torch.logical_and(
        torch.logical_not(prediction),
        truth,
    ).sum().item()

    dice_denominator = (
        2 * true_positive + false_positive + false_negative
    )
    dice = (
        2.0 * true_positive / dice_denominator
        if dice_denominator > 0
        else 1.0
    )
    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )
    recall_denominator = true_positive + false_negative
    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 1.0
    )
    predicted_percent = (
        100.0 * prediction.sum().item() / prediction.numel()
    )
    true_percent = 100.0 * truth.sum().item() / truth.numel()
    return dice, precision, recall, predicted_percent, true_percent


def _largest_lesion_slice(
    image: torch.Tensor,
    label: torch.Tensor,
    probability: torch.Tensor,
) -> VisualSlice:
    image_volume = image.squeeze().cpu().numpy()
    label_volume = label.squeeze().cpu().numpy()
    probability_volume = probability.squeeze().cpu().numpy()
    slice_index = int(np.argmax(label_volume.sum(axis=(0, 1))))
    return VisualSlice(
        image=np.rot90(image_volume[:, :, slice_index]),
        label=np.rot90(label_volume[:, :, slice_index]),
        probability=np.rot90(
            probability_volume[:, :, slice_index]
        ),
    )


def _assign_quartiles(records: list[MetricRecord]) -> tuple[float, ...]:
    case_sizes: dict[str, int] = {}
    for record in records:
        case_sizes[record.case_id] = record.full_lesion_voxels
    boundaries = tuple(
        float(value)
        for value in np.quantile(
            list(case_sizes.values()),
            [0.25, 0.5, 0.75],
        )
    )
    quartiles = {
        case_id: f"Q{np.searchsorted(boundaries, size, side='right') + 1}"
        for case_id, size in case_sizes.items()
    }
    for record in records:
        record.lesion_size_quartile = quartiles[record.case_id]
    return boundaries


def _summarize_group(
    group_type: str,
    group_value: str,
    records: list[MetricRecord],
) -> SummaryRecord:
    if not records:
        raise ValueError(f"Cannot summarize empty group {group_value}")

    threshold_scores: dict[float, float] = {}
    for threshold in THRESHOLDS:
        values = [
            record.dice
            for record in records
            if record.threshold == threshold
        ]
        if values:
            threshold_scores[threshold] = statistics.fmean(values)
    best_threshold = max(
        threshold_scores,
        key=lambda threshold: (
            threshold_scores[threshold],
            -abs(threshold - 0.5),
        ),
    )
    selected = [
        record
        for record in records
        if record.threshold == best_threshold
    ]
    return SummaryRecord(
        group_type=group_type,
        group_value=group_value,
        best_threshold=best_threshold,
        evaluation_count=len(selected),
        unique_case_count=len(
            {record.case_id for record in selected}
        ),
        mean_thresholded_dice=statistics.fmean(
            record.dice for record in selected
        ),
        mean_soft_dice=statistics.fmean(
            record.soft_dice for record in selected
        ),
        mean_precision=statistics.fmean(
            record.precision for record in selected
        ),
        mean_recall=statistics.fmean(
            record.recall for record in selected
        ),
        mean_predicted_foreground_percent=statistics.fmean(
            record.predicted_foreground_percent
            for record in selected
        ),
        mean_true_foreground_percent=statistics.fmean(
            record.true_foreground_percent for record in selected
        ),
    )


def _build_summaries(
    records: list[MetricRecord],
) -> list[SummaryRecord]:
    summaries = [_summarize_group("overall", "all", records)]

    for spec in CHECKPOINTS:
        subset = [
            record for record in records
            if record.checkpoint == spec.name
        ]
        summaries.append(
            _summarize_group("checkpoint", spec.name, subset)
        )
    for crop_size in CROP_SIZES:
        subset = [
            record for record in records
            if record.crop_size == crop_size
        ]
        summaries.append(
            _summarize_group("crop_size", str(crop_size), subset)
        )
    for lesion_class in LESION_CLASSES:
        subset = [
            record for record in records
            if record.lesion_class == lesion_class
        ]
        summaries.append(
            _summarize_group(
                "lesion_class",
                lesion_class,
                subset,
            )
        )
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        subset = [
            record for record in records
            if record.lesion_size_quartile == quartile
        ]
        summaries.append(
            _summarize_group(
                "lesion_size_quartile",
                quartile,
                subset,
            )
        )
    for spec in CHECKPOINTS:
        for crop_size in CROP_SIZES:
            subset = [
                record for record in records
                if record.checkpoint == spec.name
                and record.crop_size == crop_size
            ]
            summaries.append(
                _summarize_group(
                    "checkpoint_crop_size",
                    f"{spec.name}|{crop_size}",
                    subset,
                )
            )
    return summaries


def _write_dataclass_csv(
    path: Path,
    rows: list[MetricRecord] | list[SummaryRecord],
) -> None:
    if not rows:
        raise ValueError(f"No rows to save to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(dictionaries[0]),
        )
        writer.writeheader()
        writer.writerows(dictionaries)


def _save_figures(
    visuals: dict[tuple[str, int, str], VisualSlice],
    figure_cases: list[str],
    summary_lookup: dict[tuple[str, int], SummaryRecord],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for case_id in figure_cases:
        for crop_size in CROP_SIZES:
            figure, axes = plt.subplots(
                len(CHECKPOINTS),
                4,
                figsize=(16, 4 * len(CHECKPOINTS)),
                constrained_layout=True,
            )
            for row, spec in enumerate(CHECKPOINTS):
                visual = visuals[(case_id, crop_size, spec.name)]
                threshold = summary_lookup[
                    (spec.name, crop_size)
                ].best_threshold
                binary = visual.probability >= threshold

                axes[row, 0].imshow(visual.image, cmap="gray")
                axes[row, 0].set_title(
                    f"{spec.name}\nNormalized crop"
                )
                axes[row, 1].imshow(visual.label, cmap="gray")
                axes[row, 1].set_title("Expert mask")
                heatmap = axes[row, 2].imshow(
                    visual.probability,
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                )
                axes[row, 2].set_title("Probability")
                figure.colorbar(
                    heatmap,
                    ax=axes[row, 2],
                    fraction=0.046,
                    pad=0.04,
                )
                axes[row, 3].imshow(visual.image, cmap="gray")
                axes[row, 3].imshow(
                    np.ma.masked_where(~binary, binary),
                    cmap=PREDICTION_CMAP,
                    alpha=0.55,
                    vmin=0,
                    vmax=1,
                )
                axes[row, 3].set_title(
                    f"Binary prediction (t={threshold:.1f})"
                )
                for column in range(4):
                    axes[row, column].axis("off")

            figure.suptitle(
                f"{case_id} — oracle-centered {crop_size}³ crop\n"
                "RESEARCH USE ONLY — EXPERT-LOCALIZED, NOT CLINICAL",
                fontweight="bold",
            )
            figure.savefig(
                FIGURE_DIR / f"{case_id}_crop_{crop_size}.png",
                dpi=140,
                bbox_inches="tight",
            )
            plt.close(figure)


def _print_required_summary(
    summaries: list[SummaryRecord],
) -> None:
    print("\n" + "=" * 100)
    print("ORACLE-CROP CHECKPOINT/CROP SUMMARY — RESEARCH USE ONLY")
    print("=" * 100)
    print(
        f"{'Checkpoint':<29} {'Crop':>5} {'Best t':>7} "
        f"{'Dice':>8} {'Soft':>8} {'Prec':>8} {'Recall':>8}"
    )
    print("-" * 100)
    for summary in summaries:
        if summary.group_type != "checkpoint_crop_size":
            continue
        checkpoint, crop_size = summary.group_value.split("|")
        print(
            f"{checkpoint:<29} {crop_size:>5} "
            f"{summary.best_threshold:>7.1f} "
            f"{summary.mean_thresholded_dice:>8.4f} "
            f"{summary.mean_soft_dice:>8.4f} "
            f"{summary.mean_precision:>8.4f} "
            f"{summary.mean_recall:>8.4f}"
        )


def main() -> None:
    """Run expert-centered crop evaluation on validation cases only."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    validation_cases = build_data_list(split="val")
    models, epochs = _load_checkpoint_models(device)

    print("=" * 80)
    print("OralVision AI — RESEARCH-ONLY ORACLE-CROP EVALUATION")
    print("=" * 80)
    print(
        "WARNING: Expert labels determine crop centers. Results are an "
        "oracle localization diagnostic, not deployable performance."
    )
    print("Split: val only")
    print(f"Cases: {len(validation_cases)}")
    print(f"Device: {device}")
    print(f"Crop sizes: {CROP_SIZES}")
    print(f"Thresholds: {THRESHOLDS}")

    records: list[MetricRecord] = []
    visuals: dict[tuple[str, int, str], VisualSlice] = {}
    figure_cases = [
        case["case_id"]
        for case in validation_cases[:FIGURE_CASE_COUNT]
    ]

    for case_index, case in enumerate(validation_cases, start=1):
        case_id = case["case_id"]
        lesion_class = case["lesion_code"]
        if lesion_class not in LESION_CLASSES:
            raise ValueError(
                f"Unexpected lesion class for {case_id}: {lesion_class}"
            )
        image, label = _load_case(case)
        center, bbox_min, bbox_max = _label_bbox_center(label)
        lesion_voxels = int(label.sum())

        print(
            f"\n[{case_index:03d}/{len(validation_cases):03d}] "
            f"{case_id} ({lesion_class}) | voxels={lesion_voxels:,} | "
            f"bbox={bbox_min}..{bbox_max} | center={center}"
        )

        for crop_size in CROP_SIZES:
            image_crop, label_crop = _prepare_crop(
                image,
                label,
                center,
                crop_size,
            )
            if int(label_crop.sum().item()) == 0:
                raise ValueError(
                    f"Oracle crop lost lesion for {case_id}, "
                    f"size {crop_size}"
                )

            for spec in CHECKPOINTS:
                probabilities = _infer_probabilities(
                    models[spec.name],
                    image_crop,
                    crop_size,
                    device,
                )
                if probabilities.shape != label_crop.shape:
                    raise ValueError(
                        f"Probability/label mismatch for {case_id}, "
                        f"{spec.name}, {crop_size}"
                    )
                soft_dice = _soft_dice(
                    probabilities,
                    label_crop,
                )
                for threshold in THRESHOLDS:
                    (
                        dice,
                        precision,
                        recall,
                        predicted_percent,
                        true_percent,
                    ) = _threshold_metrics(
                        probabilities,
                        label_crop,
                        threshold,
                    )
                    values = (
                        dice,
                        soft_dice,
                        precision,
                        recall,
                        predicted_percent,
                        true_percent,
                    )
                    if not all(np.isfinite(value) for value in values):
                        raise ValueError(
                            f"Non-finite metric for {case_id}, "
                            f"{spec.name}, crop {crop_size}, "
                            f"threshold {threshold}"
                        )
                    records.append(
                        MetricRecord(
                            checkpoint=spec.name,
                            checkpoint_path=str(spec.path),
                            epoch=epochs[spec.name],
                            case_id=case_id,
                            lesion_class=lesion_class,
                            full_lesion_voxels=lesion_voxels,
                            lesion_size_quartile="",
                            crop_size=crop_size,
                            threshold=threshold,
                            dice=dice,
                            soft_dice=soft_dice,
                            precision=precision,
                            recall=recall,
                            predicted_foreground_percent=(
                                predicted_percent
                            ),
                            true_foreground_percent=true_percent,
                        )
                    )

                if case_id in figure_cases:
                    visuals[
                        (case_id, crop_size, spec.name)
                    ] = _largest_lesion_slice(
                        image_crop,
                        label_crop,
                        probabilities,
                    )

    quartile_boundaries = _assign_quartiles(records)
    summaries = _build_summaries(records)
    _write_dataclass_csv(DETAIL_CSV, records)
    _write_dataclass_csv(SUMMARY_CSV, summaries)

    summary_lookup = {
        (
            summary.group_value.split("|")[0],
            int(summary.group_value.split("|")[1]),
        ): summary
        for summary in summaries
        if summary.group_type == "checkpoint_crop_size"
    }
    _save_figures(visuals, figure_cases, summary_lookup)
    _print_required_summary(summaries)

    print(
        "\nLesion-size quartile boundaries (full-label voxels): "
        f"{quartile_boundaries}"
    )
    print(f"Detailed CSV: {DETAIL_CSV.resolve()}")
    print(f"Summary CSV: {SUMMARY_CSV.resolve()}")
    print(f"Comparison figures: {FIGURE_DIR.resolve()}")
    print(
        "RESEARCH USE ONLY — ORACLE LABEL LOCALIZATION; "
        "NOT FOR CLINICAL USE."
    )


if __name__ == "__main__":
    main()
