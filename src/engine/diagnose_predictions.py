"""Research-use-only prediction diagnostics for the epoch 2 checkpoint.

Evaluates the first five validation cases with the same full-volume
sliding-window settings as ``validate_checkpoint.py`` and saves one visual
comparison per case.

Run from the project root::

    python -m src.engine.diagnose_predictions
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference

from src.data.cached_dataset import build_data_list
from src.data.validation_transforms import get_validation_transforms
from src.engine.validate_checkpoint import (
    CHECKPOINT_PATH,
    OVERLAP,
    ROI_SIZE,
    SW_BATCH_SIZE,
    _check_input_tensors,
    _dice_score,
    _load_model,
    _single_string,
)


# Use a non-interactive backend so figure generation works on Windows without
# requiring an open desktop window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


NUM_CASES = 5
OUTPUT_DIR = Path("outputs/diagnostics_epoch_002")
THRESHOLDS: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


def _check_prediction(
    probabilities: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    case_id: str,
) -> None:
    """Verify prediction shape and numerical validity."""
    if probabilities.shape != images.shape:
        raise ValueError(
            f"Prediction/image shape mismatch for {case_id}: "
            f"{tuple(probabilities.shape)} vs {tuple(images.shape)}"
        )
    if probabilities.shape != labels.shape:
        raise ValueError(
            f"Prediction/label shape mismatch for {case_id}: "
            f"{tuple(probabilities.shape)} vs {tuple(labels.shape)}"
        )
    if not torch.isfinite(probabilities).all():
        raise ValueError(
            f"Prediction contains non-finite values: {case_id}"
        )
    if probabilities.min().item() < 0.0:
        raise ValueError(
            f"Prediction probability below zero: {case_id}"
        )
    if probabilities.max().item() > 1.0:
        raise ValueError(
            f"Prediction probability above one: {case_id}"
        )


def _display_slice(volume: np.ndarray, slice_index: int) -> np.ndarray:
    """Extract and orient one axial slice for display."""
    return np.rot90(volume[:, :, slice_index])


def _save_comparison_figure(
    *,
    image: torch.Tensor,
    label: torch.Tensor,
    probabilities: torch.Tensor,
    case_id: str,
    lesion_code: str,
) -> Path:
    """Save a four-panel comparison at the largest lesion-area slice."""
    image_volume = image.squeeze(0).squeeze(0).cpu().numpy()
    label_volume = label.squeeze(0).squeeze(0).cpu().numpy()
    probability_volume = (
        probabilities.squeeze(0).squeeze(0).cpu().numpy()
    )

    if image_volume.shape != label_volume.shape:
        raise ValueError(
            f"Visualization image/label mismatch for {case_id}: "
            f"{image_volume.shape} vs {label_volume.shape}"
        )
    if image_volume.shape != probability_volume.shape:
        raise ValueError(
            f"Visualization image/prediction mismatch for {case_id}: "
            f"{image_volume.shape} vs {probability_volume.shape}"
        )

    # Sum over the first two spatial dimensions to select the axial slice
    # with the largest expert-labelled lesion area.
    lesion_area_per_slice = label_volume.sum(axis=(0, 1))
    slice_index = int(np.argmax(lesion_area_per_slice))

    image_slice = _display_slice(image_volume, slice_index)
    label_slice = _display_slice(label_volume, slice_index)
    probability_slice = _display_slice(
        probability_volume,
        slice_index,
    )
    binary_slice = probability_slice >= 0.5

    figure, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(20, 5),
        constrained_layout=True,
    )

    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Normalized CBCT")

    axes[1].imshow(image_slice, cmap="gray")
    axes[1].imshow(
        np.ma.masked_where(label_slice == 0, label_slice),
        cmap="Reds",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Expert label overlay")

    heatmap = axes[2].imshow(
        probability_slice,
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Prediction probability")
    figure.colorbar(
        heatmap,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    axes[3].imshow(image_slice, cmap="gray")
    axes[3].imshow(
        np.ma.masked_where(~binary_slice, binary_slice),
        cmap="cool",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[3].set_title("Prediction overlay (threshold 0.5)")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{case_id} ({lesion_code}) — axial slice {slice_index}\n"
        "RESEARCH USE ONLY — NOT FOR CLINICAL USE",
        fontsize=13,
        fontweight="bold",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{case_id}_diagnostic.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _print_case_diagnostics(
    *,
    case_number: int,
    case_id: str,
    lesion_code: str,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
) -> None:
    """Print voxel statistics and threshold-specific Dice scores."""
    total_voxels = labels.numel()
    ground_truth_count = int((labels > 0).sum().item())
    predicted_mask = probabilities >= 0.5
    predicted_count = int(predicted_mask.sum().item())

    predicted_percentage = 100.0 * predicted_count / total_voxels
    ground_truth_percentage = (
        100.0 * ground_truth_count / total_voxels
    )

    print("\n" + "-" * 72)
    print(
        f"Case {case_number}/{NUM_CASES}: "
        f"{case_id} | Lesion type: {lesion_code}"
    )
    print(f"Ground-truth positive voxels: {ground_truth_count:,}")
    print(
        "Predicted probability: "
        f"min={probabilities.min().item():.6f}, "
        f"max={probabilities.max().item():.6f}, "
        f"mean={probabilities.mean().item():.6f}"
    )
    print(
        f"Predicted positive voxels at 0.5: {predicted_count:,}"
    )
    print(f"Predicted foreground: {predicted_percentage:.6f}%")
    print(
        f"Ground-truth foreground: {ground_truth_percentage:.6f}%"
    )
    print("Dice by threshold:")

    for threshold in THRESHOLDS:
        dice = _dice_score(probabilities >= threshold, labels)
        if not np.isfinite(dice):
            raise ValueError(
                f"Non-finite Dice for {case_id} at {threshold:.1f}"
            )
        print(f"  {threshold:.1f}: {dice:.6f}")


def main() -> None:
    """Run diagnostic inference for the first five validation cases."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output_device = torch.device("cpu")

    validation_data = build_data_list(split="val")
    if len(validation_data) < NUM_CASES:
        raise ValueError(
            f"Expected at least {NUM_CASES} validation cases, "
            f"found {len(validation_data)}"
        )

    dataset = Dataset(
        data=validation_data[:NUM_CASES],
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
    print("OralVision AI — RESEARCH-USE-ONLY PREDICTION DIAGNOSTICS")
    print("=" * 72)
    print("Not for clinical diagnosis or treatment decisions.")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"Cases: first {NUM_CASES} cases from the val split")
    print(f"Device: {device}")
    print(
        f"Sliding window: ROI={ROI_SIZE}, "
        f"sw_batch_size={SW_BATCH_SIZE}, overlap={OVERLAP}"
    )
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    with torch.inference_mode():
        for case_number, batch in enumerate(loader, start=1):
            case_id = _single_string(batch["case_id"], "case_id")
            lesion_code = _single_string(
                batch["lesion_code"],
                "lesion_code",
            )
            images = batch["image"]
            labels = batch["label"]
            _check_input_tensors(images, labels, case_id)

            # Full volumes and stitched predictions remain on CPU; only
            # sliding-window batches are transferred to the accelerator.
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

            if not torch.isfinite(logits).all():
                raise ValueError(
                    f"Logits contain non-finite values: {case_id}"
                )
            probabilities = torch.sigmoid(logits.float())
            _check_prediction(
                probabilities,
                images,
                labels,
                case_id,
            )

            _print_case_diagnostics(
                case_number=case_number,
                case_id=case_id,
                lesion_code=lesion_code,
                labels=labels,
                probabilities=probabilities,
            )
            output_path = _save_comparison_figure(
                image=images,
                label=labels,
                probabilities=probabilities,
                case_id=case_id,
                lesion_code=lesion_code,
            )
            print(
                f"Research-use-only figure saved: {output_path}"
            )

    print("\n" + "=" * 72)
    print("Diagnostics completed successfully.")
    print(f"Figures saved under: {OUTPUT_DIR.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
