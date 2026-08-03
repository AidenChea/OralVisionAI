from pathlib import Path
import csv

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class OralVisionDataset(Dataset):
    def __init__(
        self,
        split: str,
        dataset_root: str | Path = "data/DOLCHID",
        transform=None,
    ) -> None:
        valid_splits = {"train", "val", "test"}

        if split not in valid_splits:
            raise ValueError(
                f"split must be one of {valid_splits}, got {split!r}"
            )

        self.split = split
        self.transform = transform
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / "cbct_image"
        self.label_dir = self.dataset_root / "cbct_label"
        self.split_file = self.dataset_root / "splits.csv"

        if not self.split_file.exists():
            raise FileNotFoundError(
                f"Could not find split file: {self.split_file}"
            )

        self.cases = self._load_cases()

        if not self.cases:
            raise ValueError(f"No cases found for split {split!r}")

    def _load_cases(self) -> list[dict[str, str]]:
        cases: list[dict[str, str]] = []

        with self.split_file.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["split"] == self.split:
                    cases.append(row)

        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, object]:
        case = self.cases[index]
        case_id = case["case_id"]

        image_path = (
            self.image_dir
            / f"{case_id}_CBCT_Image.nii.gz"
        )
        label_path = (
            self.label_dir
            / f"{case_id}_CBCT_Label.nii.gz"
        )

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        image_nifti = nib.load(image_path)
        label_nifti = nib.load(label_path)

        image = image_nifti.get_fdata(dtype=np.float32)
        label = label_nifti.get_fdata(dtype=np.float32)

        if image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: "
                f"image={image.shape}, label={label.shape}"
            )

        sample = {
            "case_id": case_id,
            "lesion_code": case["lesion_code"],
            "image": image,
            "label": label,
            "spacing": image_nifti.header.get_zooms()[:3],
            "image_path": str(image_path),
            "label_path": str(label_path),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
        