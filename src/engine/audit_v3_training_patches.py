"""Audit Experiment V3 patch prevalence against the V2 8.91% baseline."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from monai.data import list_data_collate
from monai.utils import set_determinism

from src.data.cached_dataset import OralVisionCachedDataset
from src.data.cached_training_transforms_v3 import (
    DEFAULT_CACHE_DIR_V3,
    PATCH_SIZE,
    get_cached_training_transforms_v3,
)


NUM_CASES = 25
PATCHES_PER_CASE = 8
V2_MEAN_FOREGROUND_PERCENT = 8.91
EXPECTED_BATCH_SHAPE = (PATCHES_PER_CASE, 1, *PATCH_SIZE)
OUTPUT_PATH = Path("outputs/experiment_v3/patch_audit.csv")
AUDIT_SEED = 2026


@dataclass(frozen=True)
class PatchRecord:
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


def _check_patch(
    image: torch.Tensor,
    label: torch.Tensor,
    case_id: str,
    position: int,
) -> None:
    expected = (1, *PATCH_SIZE)
    if tuple(image.shape) != expected or tuple(label.shape) != expected:
        raise ValueError(
            f"Bad patch shape for {case_id} position {position}: "
            f"{tuple(image.shape)}, {tuple(label.shape)}"
        )
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError(
            f"Non-finite image for {case_id} position {position}"
        )
    if not bool(torch.isfinite(label).all().item()):
        raise ValueError(
            f"Non-finite label for {case_id} position {position}"
        )
    if not bool(torch.logical_or(label == 0, label == 1).all().item()):
        raise ValueError(
            f"Non-binary label for {case_id} position {position}"
        )


def _save(records: list[PatchRecord]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
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
        for item in records:
            writer.writerow(
                {
                    "case_id": item.case_id,
                    "lesion_code": item.lesion_code,
                    "patch_position": item.patch_position,
                    "foreground_voxel_count": (
                        item.foreground_voxel_count
                    ),
                    "foreground_percentage": (
                        f"{item.foreground_percentage:.8f}"
                    ),
                    "contains_lesion": str(
                        item.contains_lesion
                    ).lower(),
                    "image_minimum": f"{item.image_minimum:.8f}",
                    "image_maximum": f"{item.image_maximum:.8f}",
                    "image_mean": f"{item.image_mean:.8f}",
                    "image_standard_deviation": (
                        f"{item.image_standard_deviation:.8f}"
                    ),
                    "research_use_only": "true",
                }
            )


def main() -> None:
    set_determinism(seed=AUDIT_SEED)
    dataset = OralVisionCachedDataset(
        split="train",
        cache_dir=DEFAULT_CACHE_DIR_V3,
        transform=get_cached_training_transforms_v3(),
        case_indices=range(NUM_CASES),
    )
    records: list[PatchRecord] = []

    print("=" * 72)
    print("OralVision AI — EXPERIMENT V3 PATCH AUDIT")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print(f"Cases: {NUM_CASES}")
    print(f"Patches per case: {PATCHES_PER_CASE}")
    print(f"Cache: {dataset.cache_dir.resolve()}")

    for case_index in range(len(dataset)):
        case = dataset.cases[case_index]
        batch = list_data_collate([dataset[case_index]])
        images = batch["image"]
        labels = batch["label"]
        if tuple(images.shape) != EXPECTED_BATCH_SHAPE:
            raise ValueError(
                f"Bad batch shape for {case['case_id']}: "
                f"{tuple(images.shape)}"
            )

        for patch_index in range(PATCHES_PER_CASE):
            image = images[patch_index]
            label = labels[patch_index]
            position = patch_index + 1
            _check_patch(
                image,
                label,
                case["case_id"],
                position,
            )
            count = int((label > 0).sum().item())
            records.append(
                PatchRecord(
                    case_id=case["case_id"],
                    lesion_code=case["lesion_code"],
                    patch_position=position,
                    foreground_voxel_count=count,
                    foreground_percentage=(
                        100.0 * count / label.numel()
                    ),
                    contains_lesion=count > 0,
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
            f"{case['case_id']}"
        )

    expected_count = NUM_CASES * PATCHES_PER_CASE
    if len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} records, got {len(records)}"
        )

    percentages = [item.foreground_percentage for item in records]
    lesion_count = sum(item.contains_lesion for item in records)
    lesion_free_count = len(records) - lesion_count
    mean_percentage = statistics.fmean(percentages)

    print("\nPatch prevalence summary")
    print(f"Total patches: {len(records)}")
    print(f"Mean foreground percentage: {mean_percentage:.6f}%")
    print(
        f"Median foreground percentage: "
        f"{statistics.median(percentages):.6f}%"
    )
    print(f"Lesion-containing patches: {lesion_count}")
    print(f"Lesion-free patches: {lesion_free_count}")
    print(
        f"V2 reference mean foreground: "
        f"{V2_MEAN_FOREGROUND_PERCENT:.2f}%"
    )
    print(
        f"Change from V2: "
        f"{mean_percentage - V2_MEAN_FOREGROUND_PERCENT:+.6f} "
        "percentage points"
    )
    if mean_percentage >= V2_MEAN_FOREGROUND_PERCENT:
        print(
            "WARNING: V3 did not reduce mean patch foreground below V2."
        )
    else:
        print("V3 reduced mean patch foreground below V2.")

    _save(records)
    print(f"Audit CSV saved to: {OUTPUT_PATH.resolve()}")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
