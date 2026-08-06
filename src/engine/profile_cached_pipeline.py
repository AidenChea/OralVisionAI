"""Benchmark the persistent-cache training pipeline.

Runs the same cases twice:
  1. Cold cache — NIfTI load, deterministic preprocessing, and cache write
  2. Warm cache — load preprocessed tensors from disk, then random augment

Verifies patch shapes ``[1, 96, 96, 96]``, binary labels, and prints speedup.

Run from the project root::

    python -m src.engine.profile_cached_pipeline
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from monai.data import DataLoader, list_data_collate

from src.data.cached_dataset import create_cached_training_dataset
from src.data.cached_training_transforms import (
    DEFAULT_CACHE_DIR,
    PATCH_SIZE,
)
from src.data.dataset import OralVisionDataset
from src.data.training_transforms import get_training_transforms


MIN_CASES = 5
EXPECTED_PATCH_SHAPE = (1, *PATCH_SIZE)
EXPECTED_BATCH_SHAPE = (2, 1, *PATCH_SIZE)


@dataclass
class PassTimings:
    """Per-case timings accumulated over one profiling pass."""

    case_ids: list[str] = field(default_factory=list)
    durations_s: list[float] = field(default_factory=list)

    def record(self, case_id: str, duration_s: float) -> None:
        self.case_ids.append(case_id)
        self.durations_s.append(duration_s)

    @property
    def count(self) -> int:
        return len(self.durations_s)

    @property
    def total_s(self) -> float:
        return sum(self.durations_s)

    @property
    def average_s(self) -> float:
        if not self.durations_s:
            return 0.0
        return self.total_s / len(self.durations_s)


def _collate_batch(sample: dict[str, object]) -> dict[str, torch.Tensor]:
    """Wrap a multi-patch sample for ``list_data_collate``."""
    return list_data_collate([sample])


def _to_numpy(array: object) -> np.ndarray:
    """Convert tensors or arrays to ``numpy`` for validation."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def verify_patch_batch(
    batch: dict[str, object],
    case_id: str,
) -> None:
    """Ensure collated patches match training expectations."""
    images = batch["image"]
    labels = batch["label"]

    if not isinstance(images, torch.Tensor):
        raise TypeError(
            f"Expected tensor images for {case_id}, got {type(images)}"
        )
    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            f"Expected tensor labels for {case_id}, got {type(labels)}"
        )

    if tuple(images.shape) != EXPECTED_BATCH_SHAPE:
        raise ValueError(
            f"Unexpected image batch shape for {case_id}: "
            f"{tuple(images.shape)}, expected {EXPECTED_BATCH_SHAPE}"
        )
    if tuple(labels.shape) != EXPECTED_BATCH_SHAPE:
        raise ValueError(
            f"Unexpected label batch shape for {case_id}: "
            f"{tuple(labels.shape)}, expected {EXPECTED_BATCH_SHAPE}"
        )

    label_values = _to_numpy(labels)
    unique_values = np.unique(label_values)
    if not np.all(np.isin(unique_values, [0.0, 1.0])):
        raise ValueError(
            f"Non-binary label values for {case_id}: {unique_values}"
        )

    for patch_index in range(EXPECTED_BATCH_SHAPE[0]):
        image_shape = tuple(images[patch_index].shape)
        label_shape = tuple(labels[patch_index].shape)
        if image_shape != EXPECTED_PATCH_SHAPE:
            raise ValueError(
                f"Unexpected image patch shape for {case_id} "
                f"patch {patch_index}: {image_shape}"
            )
        if label_shape != EXPECTED_PATCH_SHAPE:
            raise ValueError(
                f"Unexpected label patch shape for {case_id} "
                f"patch {patch_index}: {label_shape}"
            )


def profile_cached_pass(
    dataset,
    *,
    pass_name: str,
) -> PassTimings:
    """Time ``dataset[i]`` + collate for every case in *dataset*."""
    timings = PassTimings()

    print(f"\n{'=' * 60}")
    print(pass_name)
    print("=" * 60)

    for index in range(len(dataset)):
        case_id = dataset.cases[index]["case_id"]
        start = perf_counter()
        sample = dataset[index]
        batch = _collate_batch(sample)
        duration_s = perf_counter() - start

        verify_patch_batch(batch, case_id)
        timings.record(case_id, duration_s)

        print(f"  {case_id}: {duration_s:7.3f} s")

    print(f"\n  Average: {timings.average_s:.3f} s/case")
    print(f"  Total:   {timings.total_s:.3f} s")

    return timings


def profile_uncached_baseline(
    *,
    dataset_root: Path,
    case_indices: list[int],
) -> PassTimings:
    """Time the original uncached pipeline on the same cases."""
    timings = PassTimings()
    raw_dataset = OralVisionDataset(
        split="train",
        dataset_root=dataset_root,
        transform=None,
    )
    transform = get_training_transforms()

    print(f"\n{'=' * 60}")
    print("Baseline (uncached original pipeline)")
    print("=" * 60)

    for index in case_indices:
        case = raw_dataset.cases[index]
        case_id = case["case_id"]

        start = perf_counter()
        sample = raw_dataset[index]
        transformed = transform(sample)
        batch = list_data_collate([transformed])
        duration_s = perf_counter() - start

        verify_patch_batch(batch, case_id)
        timings.record(case_id, duration_s)

        print(f"  {case_id}: {duration_s:7.3f} s")

    print(f"\n  Average: {timings.average_s:.3f} s/case")
    print(f"  Total:   {timings.total_s:.3f} s")

    return timings


def _print_speedup_report(
    *,
    cold: PassTimings,
    warm: PassTimings,
    baseline: PassTimings | None,
) -> None:
    """Print comparative speedup summaries."""
    cache_speedup = (
        cold.average_s / warm.average_s
        if warm.average_s > 0
        else float("inf")
    )

    print(f"\n{'=' * 60}")
    print("Speedup summary")
    print("=" * 60)
    print(f"Cold cache average:  {cold.average_s:.3f} s/case")
    print(f"Warm cache average:  {warm.average_s:.3f} s/case")
    print(
        f"Warm vs cold speedup: {cache_speedup:.2f}x "
        f"({cold.average_s - warm.average_s:.3f} s saved/case)"
    )

    if baseline is not None and warm.average_s > 0:
        total_speedup = baseline.average_s / warm.average_s
        print(f"Baseline average:    {baseline.average_s:.3f} s/case")
        print(
            f"Warm vs baseline:    {total_speedup:.2f}x "
            f"({baseline.average_s - warm.average_s:.3f} s saved/case)"
        )

        projected_epoch_min = warm.average_s * 183 / 60
        baseline_epoch_min = baseline.average_s * 183 / 60
        print(
            f"\nProjected epoch (183 cases, warm cache): "
            f"{projected_epoch_min:.1f} min"
        )
        print(
            f"Original pipeline epoch estimate: "
            f"{baseline_epoch_min:.1f} min"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the persistent-cache OralVisionAI training pipeline."
        ),
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=MIN_CASES,
        help=(
            f"Number of training cases to profile "
            f"(default: {MIN_CASES}, minimum: {MIN_CASES})"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/DOLCHID"),
        help="Root directory of the CBCT dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(DEFAULT_CACHE_DIR),
        help="Persistent cache directory for deterministic preprocessing.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help=(
            "Do not delete the cache directory before profiling. "
            "By default the cache is cleared so pass 1 measures "
            "true cold-cache creation time."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the uncached baseline comparison (saves time).",
    )
    args = parser.parse_args()

    num_cases = max(args.num_cases, MIN_CASES)
    case_indices = list(range(num_cases))

    if not args.keep_cache and args.cache_dir.exists():
        print(f"Clearing cache directory: {args.cache_dir}")
        shutil.rmtree(args.cache_dir)

    dataset = create_cached_training_dataset(
        split="train",
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        case_indices=case_indices,
    )

    # Smoke-test DataLoader wiring with num_workers=0 (Windows-safe).
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    smoke_batch = next(iter(loader))
    verify_patch_batch(
        smoke_batch,
        dataset.cases[0]["case_id"],
    )

    print("=" * 60)
    print("OralVision AI — Cached Pipeline Profiler")
    print("=" * 60)
    print(f"Cases profiled: {num_cases}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Patch shape: {EXPECTED_PATCH_SHAPE}")
    print(f"DataLoader smoke test: batch shape {tuple(smoke_batch['image'].shape)}")

    cold_timings = profile_cached_pass(
        dataset,
        pass_name="Pass 1 — cold cache (create + random augment)",
    )
    warm_timings = profile_cached_pass(
        dataset,
        pass_name="Pass 2 — warm cache (load cache + random augment)",
    )

    baseline_timings: PassTimings | None = None
    if not args.skip_baseline:
        baseline_timings = profile_uncached_baseline(
            dataset_root=args.dataset_root,
            case_indices=case_indices,
        )

    _print_speedup_report(
        cold=cold_timings,
        warm=warm_timings,
        baseline=baseline_timings,
    )

    print("\nAll correctness checks passed.")


if __name__ == "__main__":
    main()
