"""Overfit a fresh OralVisionAI v2 model on one immutable lesion patch.

This research-use-only diagnostic tests whether the model, loss, optimizer,
and mixed-precision path can memorize a single training example. Random v2
transforms are used only to select the patch; the resulting tensors are then
cloned and reused unchanged for every optimization step.

Run from the project root::

    python -m src.engine.overfit_single_patch
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from monai.data import list_data_collate
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from src.data.cached_dataset import OralVisionCachedDataset
from src.data.cached_training_transforms_v2 import (
    DEFAULT_CACHE_DIR_V2,
    PATCH_SIZE,
    get_cached_training_transforms_v2,
)
from src.models.unet3d import create_unet3d


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


MAX_STEPS = 500
PRINT_INTERVAL = 25
EARLY_STOP_DICE = 0.95
MAX_CASES_TO_SEARCH = 25
SELECTION_SEED = 2026
EXPECTED_FIXED_SHAPE = (1, 1, *PATCH_SIZE)
OUTPUT_PATH = Path(
    "outputs/experiment_v2/single_patch_overfit.png"
)
PREDICTION_CMAP = ListedColormap(["cyan"])


def _check_fixed_patch(
    image: torch.Tensor,
    label: torch.Tensor,
) -> None:
    """Verify fixed patch shape, finite values, and label correctness."""
    if tuple(image.shape) != EXPECTED_FIXED_SHAPE:
        raise ValueError(
            f"Unexpected fixed image shape: {tuple(image.shape)}; "
            f"expected {EXPECTED_FIXED_SHAPE}"
        )
    if tuple(label.shape) != EXPECTED_FIXED_SHAPE:
        raise ValueError(
            f"Unexpected fixed label shape: {tuple(label.shape)}; "
            f"expected {EXPECTED_FIXED_SHAPE}"
        )
    if image.shape != label.shape:
        raise ValueError(
            f"Image/label shape mismatch: "
            f"{tuple(image.shape)} vs {tuple(label.shape)}"
        )
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError("Fixed image contains non-finite values.")
    if not bool(torch.isfinite(label).all().item()):
        raise ValueError("Fixed label contains non-finite values.")
    if not bool(torch.logical_or(label == 0, label == 1).all().item()):
        raise ValueError(
            f"Fixed label is not binary: {torch.unique(label).tolist()}"
        )
    if int((label > 0).sum().item()) == 0:
        raise ValueError("Selected fixed patch contains no lesion.")


def _select_fixed_lesion_patch(
) -> tuple[torch.Tensor, torch.Tensor, str, str, int]:
    """Select one lesion patch with exact v2 transforms, then clone it."""
    set_determinism(seed=SELECTION_SEED)
    dataset = OralVisionCachedDataset(
        split="train",
        cache_dir=DEFAULT_CACHE_DIR_V2,
        transform=get_cached_training_transforms_v2(),
        case_indices=range(MAX_CASES_TO_SEARCH),
    )

    for case_index in range(len(dataset)):
        case = dataset.cases[case_index]
        transformed = dataset[case_index]
        batch = list_data_collate([transformed])
        images = batch["image"]
        labels = batch["label"]

        expected_batch_shape = (4, 1, *PATCH_SIZE)
        if tuple(images.shape) != expected_batch_shape:
            raise ValueError(
                f"Unexpected sampled image shape for "
                f"{case['case_id']}: {tuple(images.shape)}"
            )
        if tuple(labels.shape) != expected_batch_shape:
            raise ValueError(
                f"Unexpected sampled label shape for "
                f"{case['case_id']}: {tuple(labels.shape)}"
            )

        for patch_index in range(labels.shape[0]):
            if int((labels[patch_index] > 0).sum().item()) > 0:
                # Clone one batch-shaped pair. No dataset or random transform
                # is accessed again after this selection.
                fixed_image = (
                    images[patch_index : patch_index + 1]
                    .clone()
                    .contiguous()
                )
                fixed_label = (
                    labels[patch_index : patch_index + 1]
                    .clone()
                    .contiguous()
                )
                _check_fixed_patch(fixed_image, fixed_label)
                return (
                    fixed_image,
                    fixed_label,
                    case["case_id"],
                    case["lesion_code"],
                    patch_index + 1,
                )

    raise RuntimeError(
        f"No lesion-containing patch found in the first "
        f"{MAX_CASES_TO_SEARCH} training cases."
    )


def _dice_score(
    prediction: torch.Tensor,
    label: torch.Tensor,
) -> float:
    """Compute binary Dice for a fixed patch."""
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


def _prediction_metrics(
    logits: torch.Tensor,
    label: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Return Dice, predicted/true foreground, and mean probability."""
    if logits.shape != label.shape:
        raise ValueError(
            f"Logit/label shape mismatch: "
            f"{tuple(logits.shape)} vs {tuple(label.shape)}"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("Model logits contain non-finite values.")

    probabilities = torch.sigmoid(logits.float())
    if not bool(torch.isfinite(probabilities).all().item()):
        raise ValueError("Probabilities contain non-finite values.")

    prediction = probabilities >= 0.5
    dice = _dice_score(prediction, label)
    predicted_percent = (
        100.0 * prediction.sum().item() / prediction.numel()
    )
    true_percent = (
        100.0 * (label > 0).sum().item() / label.numel()
    )
    mean_probability = float(probabilities.mean().item())
    return dice, predicted_percent, true_percent, mean_probability


def _display_slice(volume: np.ndarray, index: int) -> np.ndarray:
    """Extract and orient one axial patch slice."""
    return np.rot90(volume[:, :, index])


def _save_final_overlay(
    *,
    image: torch.Tensor,
    label: torch.Tensor,
    probabilities: torch.Tensor,
    case_id: str,
    lesion_code: str,
    final_dice: float,
) -> None:
    """Save image, expert overlay, and final prediction overlay."""
    image_volume = image.squeeze().detach().cpu().numpy()
    label_volume = label.squeeze().detach().cpu().numpy()
    probability_volume = (
        probabilities.squeeze().detach().cpu().numpy()
    )

    if not (
        image_volume.shape
        == label_volume.shape
        == probability_volume.shape
    ):
        raise ValueError(
            "Overlay volumes do not have matching shapes: "
            f"{image_volume.shape}, {label_volume.shape}, "
            f"{probability_volume.shape}"
        )

    lesion_area = label_volume.sum(axis=(0, 1))
    slice_index = int(np.argmax(lesion_area))
    image_slice = _display_slice(image_volume, slice_index)
    label_slice = _display_slice(label_volume, slice_index)
    probability_slice = _display_slice(
        probability_volume,
        slice_index,
    )
    prediction_slice = probability_slice >= 0.5

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        constrained_layout=True,
    )
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Fixed normalized CBCT patch")

    axes[1].imshow(image_slice, cmap="gray")
    axes[1].imshow(
        np.ma.masked_where(label_slice == 0, label_slice),
        cmap="Reds",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Expert label overlay")

    axes[2].imshow(image_slice, cmap="gray")
    axes[2].imshow(
        np.ma.masked_where(
            ~prediction_slice,
            prediction_slice,
        ),
        cmap=PREDICTION_CMAP,
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Final prediction overlay (0.5)")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{case_id} ({lesion_code}) | Dice={final_dice:.4f}\n"
        "RESEARCH USE ONLY — NOT FOR CLINICAL USE",
        fontsize=13,
        fontweight="bold",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the fixed single-patch memorization test."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    (
        fixed_image,
        fixed_label,
        case_id,
        lesion_code,
        patch_position,
    ) = _select_fixed_lesion_patch()
    _check_fixed_patch(fixed_image, fixed_label)

    true_count = int((fixed_label > 0).sum().item())
    true_percent = 100.0 * true_count / fixed_label.numel()

    print("=" * 72)
    print("OralVision AI — EXPERIMENT V2 SINGLE-PATCH OVERFIT TEST")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Selected case: {case_id} ({lesion_code})")
    print(f"Selected patch position: {patch_position}")
    print(f"Fixed patch shape: {tuple(fixed_image.shape)}")
    print(f"Ground-truth positive voxels: {true_count:,}")
    print(f"True foreground percentage: {true_percent:.6f}%")
    print("Random transforms disabled after patch selection.")

    images = fixed_image.to(device, non_blocking=True)
    labels = fixed_label.to(device, non_blocking=True)

    # This is deliberately a fresh model with the exact v2 objective.
    model = create_unet3d().to(device)
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
        lambda_dice=1.0,
        lambda_ce=2.0,
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

    final_step = 0
    final_dice = 0.0
    model.train()

    for step in range(1, MAX_STEPS + 1):
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
                f"Non-finite loss at step {step}: {loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        (
            dice,
            predicted_percent,
            current_true_percent,
            mean_probability,
        ) = _prediction_metrics(logits.detach(), labels)
        final_step = step
        final_dice = dice

        should_print = (
            step % PRINT_INTERVAL == 0
            or dice > EARLY_STOP_DICE
            or step == MAX_STEPS
        )
        if should_print:
            print(
                f"Step {step:03d} | "
                f"Loss={loss.item():.6f} | "
                f"Dice@0.5={dice:.6f} | "
                f"Pred FG={predicted_percent:.6f}% | "
                f"True FG={current_true_percent:.6f}% | "
                f"Mean probability={mean_probability:.6f}"
            )

        if dice > EARLY_STOP_DICE:
            print(
                f"Early stopping: Dice {dice:.6f} exceeded "
                f"{EARLY_STOP_DICE:.2f}."
            )
            break

    model.eval()
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            final_logits = model(images)
        if not bool(torch.isfinite(final_logits).all().item()):
            raise ValueError("Final logits contain non-finite values.")
        final_probabilities = torch.sigmoid(final_logits.float())
        final_dice, _, _, _ = _prediction_metrics(
            final_logits,
            labels,
        )

    _save_final_overlay(
        image=images,
        label=labels,
        probabilities=final_probabilities,
        case_id=case_id,
        lesion_code=lesion_code,
        final_dice=final_dice,
    )

    print("\n" + "=" * 72)
    print(f"Completed optimization steps: {final_step}")
    print(f"Final evaluation-mode Dice@0.5: {final_dice:.6f}")
    print(f"Final overlay saved to: {OUTPUT_PATH.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
