"""Profile each stage of the OralVisionAI training pipeline.

Measures wall-clock time for:
  1. NIfTI load and decompression
  2. MONAI transforms and patch sampling
  3. Host-to-GPU tensor transfer
  4. Model forward pass
  5. Loss calculation and backward pass
  6. Optimizer step

Run from the project root::

    python -m src.engine.profile_training_pipeline

This script does not modify the existing training loop; it reuses the same
dataset paths, transforms, model, loss, and optimizer settings as
``train_one_epoch.py`` while timing each stage in isolation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import torch
from monai.data import list_data_collate
from monai.losses import DiceCELoss

from src.data.dataset import OralVisionDataset
from src.data.training_transforms import get_training_transforms
from src.models.unet3d import create_unet3d


# Minimum number of batches required by the profiling spec.
MIN_BATCHES = 5


@dataclass
class StageTimings:
    """Accumulated wall-clock seconds for one pipeline stage."""

    nifti_load_s: float = 0.0
    transforms_s: float = 0.0
    gpu_transfer_s: float = 0.0
    forward_s: float = 0.0
    loss_backward_s: float = 0.0
    optimizer_step_s: float = 0.0
    batch_count: int = 0

    def record(
        self,
        *,
        nifti_load_s: float,
        transforms_s: float,
        gpu_transfer_s: float,
        forward_s: float,
        loss_backward_s: float,
        optimizer_step_s: float,
    ) -> None:
        self.nifti_load_s += nifti_load_s
        self.transforms_s += transforms_s
        self.gpu_transfer_s += gpu_transfer_s
        self.forward_s += forward_s
        self.loss_backward_s += loss_backward_s
        self.optimizer_step_s += optimizer_step_s
        self.batch_count += 1

    def average(self, total_s: float) -> float:
        if self.batch_count == 0:
            return 0.0
        return total_s / self.batch_count


def _sync_cuda(device: torch.device) -> None:
    """Block until queued GPU work finishes so timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _resolve_case_paths(
    dataset: OralVisionDataset,
    index: int,
) -> tuple[str, Path, Path]:
    """Return case metadata and NIfTI paths for a dataset index."""
    case = dataset.cases[index]
    case_id = case["case_id"]

    image_path = (
        dataset.image_dir / f"{case_id}_CBCT_Image.nii.gz"
    )
    label_path = (
        dataset.label_dir / f"{case_id}_CBCT_Label.nii.gz"
    )

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    return case_id, image_path, label_path


def _load_nifti_pair(
    image_path: Path,
    label_path: Path,
    case_id: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Load and decompress image/label volumes (stage 1 logic)."""
    image_nifti = nib.load(image_path)
    label_nifti = nib.load(label_path)

    image = image_nifti.get_fdata(dtype=np.float32)
    label = label_nifti.get_fdata(dtype=np.float32)

    if image.shape != label.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}: "
            f"image={image.shape}, label={label.shape}"
        )

    spacing = image_nifti.header.get_zooms()[:3]
    return image, label, spacing


def _build_sample(
    *,
    case_id: str,
    lesion_code: str,
    image: np.ndarray,
    label: np.ndarray,
    spacing: tuple[float, ...],
    image_path: Path,
    label_path: Path,
) -> dict[str, object]:
    """Build the dict consumed by MONAI transforms."""
    return {
        "case_id": case_id,
        "lesion_code": lesion_code,
        "image": image,
        "label": label,
        "spacing": spacing,
        "image_path": str(image_path),
        "label_path": str(label_path),
    }


def profile_one_batch(
    *,
    dataset: OralVisionDataset,
    case_index: int,
    transform,
    device: torch.device,
    model: torch.nn.Module,
    loss_function: DiceCELoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> StageTimings:
    """Time every pipeline stage for a single training batch."""
    case_id, image_path, label_path = _resolve_case_paths(
        dataset,
        case_index,
    )
    case = dataset.cases[case_index]

    timings = StageTimings()

    # --- Stage 1: NIfTI load and decompression ---
    t0 = perf_counter()
    image, label, spacing = _load_nifti_pair(
        image_path,
        label_path,
        case_id,
    )
    nifti_load_s = perf_counter() - t0

    sample = _build_sample(
        case_id=case_id,
        lesion_code=case["lesion_code"],
        image=image,
        label=label,
        spacing=spacing,
        image_path=image_path,
        label_path=label_path,
    )

    # --- Stage 2: MONAI transforms and patch sampling ---
    t0 = perf_counter()
    transformed = transform(sample)
    batch = list_data_collate([transformed])
    transforms_s = perf_counter() - t0

    images = batch["image"]
    labels = batch["label"]

    # Sanity checks mirroring the dataloader tests.
    if images.shape != (2, 1, 96, 96, 96):
        raise ValueError(
            f"Unexpected image batch shape for {case_id}: "
            f"{images.shape}"
        )
    if labels.shape != (2, 1, 96, 96, 96):
        raise ValueError(
            f"Unexpected label batch shape for {case_id}: "
            f"{labels.shape}"
        )

    # --- Stage 3: Host-to-GPU transfer ---
    _sync_cuda(device)
    t0 = perf_counter()
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    _sync_cuda(device)
    gpu_transfer_s = perf_counter() - t0

    optimizer.zero_grad(set_to_none=True)

    # --- Stage 4: Forward pass ---
    _sync_cuda(device)
    t0 = perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        predictions = model(images)
    _sync_cuda(device)
    forward_s = perf_counter() - t0

    # --- Stage 5: Loss and backward pass ---
    _sync_cuda(device)
    t0 = perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        loss = loss_function(predictions, labels)

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss for case {case_id}: {loss.item()}"
        )

    scaler.scale(loss).backward()
    _sync_cuda(device)
    loss_backward_s = perf_counter() - t0

    # --- Stage 6: Optimizer step ---
    _sync_cuda(device)
    t0 = perf_counter()
    scaler.step(optimizer)
    scaler.update()
    _sync_cuda(device)
    optimizer_step_s = perf_counter() - t0

    timings.record(
        nifti_load_s=nifti_load_s,
        transforms_s=transforms_s,
        gpu_transfer_s=gpu_transfer_s,
        forward_s=forward_s,
        loss_backward_s=loss_backward_s,
        optimizer_step_s=optimizer_step_s,
    )
    return timings


def _print_report(
    totals: StageTimings,
    device: torch.device,
) -> None:
    """Print average per-stage timings and their share of total time."""
    count = totals.batch_count
    if count == 0:
        print("No batches profiled.")
        return

    averages = {
        "NIfTI load + decompress": totals.average(totals.nifti_load_s),
        "MONAI transforms + patches": totals.average(totals.transforms_s),
        "GPU transfer": totals.average(totals.gpu_transfer_s),
        "Forward pass": totals.average(totals.forward_s),
        "Loss + backward": totals.average(totals.loss_backward_s),
        "Optimizer step": totals.average(totals.optimizer_step_s),
    }

    avg_total = sum(averages.values())

    print("\n" + "=" * 60)
    print("Average timing per batch (seconds)")
    print("=" * 60)
    print(f"{'Stage':<32} {'Avg (s)':>10} {'Share':>8}")
    print("-" * 60)

    for stage_name, avg_s in averages.items():
        share_pct = (avg_s / avg_total * 100.0) if avg_total else 0.0
        print(f"{stage_name:<32} {avg_s:>10.3f} {share_pct:>7.1f}%")

    print("-" * 60)
    print(f"{'Total per batch':<32} {avg_total:>10.3f} {'100.0%':>8}")
    print(f"\nBatches profiled: {count}")
    print(
        f"Projected epoch time at this average "
        f"(183 batches): {avg_total * 183 / 60:.1f} min"
    )

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory allocated: {peak_gb:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile OralVisionAI training pipeline stages."
        ),
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=MIN_BATCHES,
        help=(
            f"Number of batches to profile "
            f"(default: {MIN_BATCHES}, minimum: {MIN_BATCHES})"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/DOLCHID"),
        help="Root directory of the CBCT dataset.",
    )
    args = parser.parse_args()

    num_batches = max(args.num_batches, MIN_BATCHES)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Case list only — transforms are applied manually so we can
    # time loading and augmentation separately.
    dataset = OralVisionDataset(
        split="train",
        dataset_root=args.dataset_root,
        transform=None,
    )
    transform = get_training_transforms()

    model = create_unet3d().to(device)
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
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

    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    totals = StageTimings()

    print("=" * 60)
    print("OralVision AI — Training Pipeline Profiler")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Training cases available: {len(dataset)}")
    print(f"Batches to profile: {num_batches}")

    for batch_idx in range(num_batches):
        case_index = batch_idx % len(dataset)
        case_id = dataset.cases[case_index]["case_id"]

        batch_timings = profile_one_batch(
            dataset=dataset,
            case_index=case_index,
            transform=transform,
            device=device,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            scaler=scaler,
        )

        totals.nifti_load_s += batch_timings.nifti_load_s
        totals.transforms_s += batch_timings.transforms_s
        totals.gpu_transfer_s += batch_timings.gpu_transfer_s
        totals.forward_s += batch_timings.forward_s
        totals.loss_backward_s += batch_timings.loss_backward_s
        totals.optimizer_step_s += batch_timings.optimizer_step_s
        totals.batch_count += 1

        batch_total = (
            batch_timings.nifti_load_s
            + batch_timings.transforms_s
            + batch_timings.gpu_transfer_s
            + batch_timings.forward_s
            + batch_timings.loss_backward_s
            + batch_timings.optimizer_step_s
        )

        print(
            f"\nBatch {batch_idx + 1}/{num_batches} "
            f"(case {case_id})"
        )
        print(
            f"  NIfTI load:     {batch_timings.nifti_load_s:6.3f} s"
        )
        print(
            f"  Transforms:     {batch_timings.transforms_s:6.3f} s"
        )
        print(
            f"  GPU transfer:   {batch_timings.gpu_transfer_s:6.3f} s"
        )
        print(
            f"  Forward:        {batch_timings.forward_s:6.3f} s"
        )
        print(
            f"  Loss+backward:  "
            f"{batch_timings.loss_backward_s:6.3f} s"
        )
        print(
            f"  Optimizer:      "
            f"{batch_timings.optimizer_step_s:6.3f} s"
        )
        print(f"  Total:          {batch_total:6.3f} s")

    _print_report(totals, device)


if __name__ == "__main__":
    main()
