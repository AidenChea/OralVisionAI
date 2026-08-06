"""MONAI ``PersistentDataset`` wrapper for cached OralVisionAI training.

Builds a file-path data list from ``splits.csv``, then delegates to
``PersistentDataset`` so deterministic preprocessing is written once to
``data/cache/training`` and random patch sampling runs on every access.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from monai.data import PersistentDataset
from monai.data.utils import pickle_hashing

from src.data.cached_training_transforms import (
    DEFAULT_CACHE_DIR,
    get_cached_training_transforms,
)


VALID_SPLITS = frozenset({"train", "val", "test"})


def load_split_cases(
    split: str,
    dataset_root: str | Path = "data/DOLCHID",
) -> list[dict[str, str]]:
    """Load case metadata rows for *split* from ``splits.csv``."""
    if split not in VALID_SPLITS:
        raise ValueError(
            f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}"
        )

    dataset_root = Path(dataset_root)
    split_file = dataset_root / "splits.csv"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Could not find split file: {split_file}"
        )

    cases: list[dict[str, str]] = []
    with split_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["split"] == split:
                cases.append(row)

    if not cases:
        raise ValueError(f"No cases found for split {split!r}")

    return cases


def build_data_list(
    split: str,
    dataset_root: str | Path = "data/DOLCHID",
    case_indices: Sequence[int] | None = None,
) -> list[dict[str, str]]:
    """Build MONAI-compatible dicts with NIfTI paths and case metadata."""
    cases = load_split_cases(split, dataset_root)
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / "cbct_image"
    label_dir = dataset_root / "cbct_label"

    if case_indices is not None:
        selected = [cases[index] for index in case_indices]
    else:
        selected = cases

    data_list: list[dict[str, str]] = []
    for case in selected:
        case_id = case["case_id"]
        image_path = image_dir / f"{case_id}_CBCT_Image.nii.gz"
        label_path = label_dir / f"{case_id}_CBCT_Label.nii.gz"

        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not label_path.exists():
            raise FileNotFoundError(label_path)

        data_list.append(
            {
                "case_id": case_id,
                "lesion_code": case["lesion_code"],
                "image": str(image_path),
                "label": str(label_path),
            }
        )

    return data_list


class OralVisionCachedDataset:
    """Training dataset with persistent deterministic preprocessing cache."""

    def __init__(
        self,
        split: str,
        dataset_root: str | Path = "data/DOLCHID",
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        transform=None,
        case_indices: Sequence[int] | None = None,
    ) -> None:
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.cache_dir = Path(cache_dir)
        self.case_indices = case_indices
        self.data_list = build_data_list(
            split=split,
            dataset_root=dataset_root,
            case_indices=case_indices,
        )
        self.transform = (
            transform
            if transform is not None
            else get_cached_training_transforms()
        )

        self._dataset = PersistentDataset(
            data=self.data_list,
            transform=self.transform,
            cache_dir=self.cache_dir,
            hash_transform=pickle_hashing,
        )

    @property
    def cases(self) -> list[dict[str, str]]:
        """Case metadata aligned with dataset indices."""
        return self.data_list

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._dataset[index]


def create_cached_training_dataset(
    split: str = "train",
    dataset_root: str | Path = "data/DOLCHID",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    case_indices: Sequence[int] | None = None,
) -> OralVisionCachedDataset:
    """Factory for a cached training dataset."""
    return OralVisionCachedDataset(
        split=split,
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        case_indices=case_indices,
    )
