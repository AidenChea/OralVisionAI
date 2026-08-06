"""Audit the exact patches sampled by OralVisionAI Experiment V2.

This research-use-only diagnostic samples the first 25 training cases once,
records all four patches per case, and summarizes the actual foreground and
image-intensity distributions seen by the model.

Run from the project root::

    python -m src.engine.audit_v2_training_patches
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from monai.data import list_data_collate
from monai.utils import set_determinism

from src.data.cached_dataset import OralVisionCachedDataset
from src.data.cached_training_transforms_v2 import (
    DEFAULT_CACHE_DIR_V2,
    PATCH_SIZE,
    get_cached_training_transforms_v2,
)


NUM_CASES = 25
PATCHES_PER_CASE = 4
EXPECTED_PATCH_SHAPE = (1, *PATCH_SIZE)
EXPECTED_BATCH_SHAPE = (PATCHES_PER_CASE, 1, *PATCH_SIZE)
OUTPUT_PATH = Path("outputs/experiment_v2/patch_audit.csv")
AUDIT_SEED = 2026


@dataclass(frozen=True)
class PatchRecord:
    """Measured properties of one sampled training patch."""

    case_id: str
    lesion_code: str
    patch_position: int
    foreground_voxel_count: int
    foreground_percentage: float
    contains_lesion: bool
    image_minimum: float
    image_maximum: float
    image_mean: float
    image_standard_deviation: float


def _validate_patch(
    image: torch.Tensor,
    label: torch.Tensor,
    case_id: str,
    patch_position: int,
) -> None:
    """Verify one patch's shape, finite values, and binary label."""
    if tuple(image.shape) != EXPECTED_PATCH_SHAPE:
        raise ValueError(
            f"Unexpected image shape for {case_id} patch "
            f"{patch_position}: {tuple(image.shape)}"
        )
    if tuple(label.shape) != EXPECTED_PATCH_SHAPE:
        raise ValueError(
            f"Unexpected label shape for {case_id} patch "
            f"{patch_position}: {tuple(label.shape)}"
        )
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError(
            f"Non-finite image values for {case_id} "
            f"patch {patch_position}"
        )
    if not bool(torch.isfinite(label).all().item()):
        raise ValueError(
            f"Non-finite label values for {case_id} "
            f"patch {patch_position}"
        )
    if not bool(torch.logical_or(label == 0, label == 1).all().item()):
        raise ValueError(
            f"Non-binary label for {case_id} patch {patch_position}: "
            f"{torch.unique(label).tolist()}"
        )


def _summarize(name: str, records: list[PatchRecord]) -> None:
    """Print foreground statistics for a collection of patches."""
    if not records:
        print(f"{name}: no patches")
        return

    percentages = [
        record.foreground_percentage for record in records
    ]
    lesion_count = sum(record.contains_lesion for record in records)
    lesion_free_count = len(records) - lesion_count

    print(f"\n{name}")
    print(f"  Total patches: {len(records)}")
    print(
        f"  Mean foreground percentage: "
        f"{statistics.fmean(percentages):.6f}%"
    )
    print(
        f"  Median foreground percentage: "
        f"{statistics.median(percentages):.6f}%"
    )
    print(
        f"  Minimum foreground percentage: "
        f"{min(percentages):.6f}%"
    )
    print(
        f"  Maximum foreground percentage: "
        f"{max(percentages):.6f}%"
    )
    print(f"  Completely lesion-free patches: {lesion_free_count}")
    print(f"  Patches containing lesion: {lesion_count}")


def _save_csv(records: list[PatchRecord]) -> None:
    """Save all patch-level measurements."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "lesion_code",
                "patch_position",
                "foreground_voxel_count",
                "foreground_percentage",
                "contains_lesion",
                "image_minimum",
                "image_maximum",
                "image_mean",
                "image_standard_deviation",
                "research_use_only",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "case_id": record.case_id,
                    "lesion_code": record.lesion_code,
                    "patch_position": record.patch_position,
                    "foreground_voxel_count": (
                        record.foreground_voxel_count
                    ),
                    "foreground_percentage": (
                        f"{record.foreground_percentage:.8f}"
                    ),
                    "contains_lesion": str(
                        record.contains_lesion
                    ).lower(),
                    "image_minimum": f"{record.image_minimum:.8f}",
                    "image_maximum": f"{record.image_maximum:.8f}",
                    "image_mean": f"{record.image_mean:.8f}",
                    "image_standard_deviation": (
                        f"{record.image_standard_deviation:.8f}"
                    ),
                    "research_use_only": "true",
                }
            )


def main() -> None:
    """Sample and audit 100 v2 training patches."""
    # Fix only the diagnostic's random draw so repeated audits are comparable.
    set_determinism(seed=AUDIT_SEED)
    dataset = OralVisionCachedDataset(
        split="train",
        cache_dir=DEFAULT_CACHE_DIR_V2,
        transform=get_cached_training_transforms_v2(),
        case_indices=range(NUM_CASES),
    )

    if len(dataset) != NUM_CASES:
        raise ValueError(
            f"Expected {NUM_CASES} cases, found {len(dataset)}"
        )

    print("=" * 72)
    print("OralVision AI — EXPERIMENT V2 TRAINING PATCH AUDIT")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print(f"Cases: {NUM_CASES}")
    print(f"Patches per case: {PATCHES_PER_CASE}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Cache: {dataset.cache_dir.resolve()}")
    print(f"Audit seed: {AUDIT_SEED}")

    records: list[PatchRecord] = []
    for case_index in range(len(dataset)):
        case = dataset.cases[case_index]
        case_id = case["case_id"]
        lesion_code = case["lesion_code"]

        transformed = dataset[case_index]
        batch = list_data_collate([transformed])
        images = batch["image"]
        labels = batch["label"]

        if tuple(images.shape) != EXPECTED_BATCH_SHAPE:
            raise ValueError(
                f"Unexpected image batch shape for {case_id}: "
                f"{tuple(images.shape)}"
            )
        if tuple(labels.shape) != EXPECTED_BATCH_SHAPE:
            raise ValueError(
                f"Unexpected label batch shape for {case_id}: "
                f"{tuple(labels.shape)}"
            )

        for patch_index in range(PATCHES_PER_CASE):
            patch_position = patch_index + 1
            image = images[patch_index]
            label = labels[patch_index]
            _validate_patch(
                image,
                label,
                case_id,
                patch_position,
            )

            foreground_count = int((label > 0).sum().item())
            foreground_percentage = (
                100.0 * foreground_count / label.numel()
            )
            records.append(
                PatchRecord(
                    case_id=case_id,
                    lesion_code=lesion_code,
                    patch_position=patch_position,
                    foreground_voxel_count=foreground_count,
                    foreground_percentage=foreground_percentage,
                    contains_lesion=foreground_count > 0,
                    image_minimum=float(image.min().item()),
                    image_maximum=float(image.max().item()),
                    image_mean=float(image.mean().item()),
                    image_standard_deviation=float(
                        image.std(unbiased=False).item()
                    ),
                )
            )

        print(
            f"[{case_index + 1:02d}/{NUM_CASES:02d}] "
            f"Audited {case_id} ({lesion_code})"
        )

    expected_total = NUM_CASES * PATCHES_PER_CASE
    if len(records) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} patch records, "
            f"created {len(records)}"
        )

    _summarize("All patch positions", records)
    for patch_position in range(1, PATCHES_PER_CASE + 1):
        position_records = [
            record
            for record in records
            if record.patch_position == patch_position
        ]
        _summarize(
            f"Patch position {patch_position}",
            position_records,
        )

    _save_csv(records)
    print(f"\nAudit CSV saved to: {OUTPUT_PATH.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
