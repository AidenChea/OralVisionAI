import csv
import random
from collections import defaultdict
from pathlib import Path


IMAGE_DIR = Path("data/DOLCHID/cbct_image")
OUTPUT_PATH = Path("data/DOLCHID/splits.csv")
RANDOM_SEED = 42


def get_case_id(path: Path) -> str:
    return path.name.replace("_CBCT_Image.nii.gz", "")


def split_group(case_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    random.shuffle(case_ids)

    total = len(case_ids)
    train_end = round(total * 0.70)
    val_end = train_end + round(total * 0.15)

    train = case_ids[:train_end]
    val = case_ids[train_end:val_end]
    test = case_ids[val_end:]

    return train, val, test


def main() -> None:
    random.seed(RANDOM_SEED)

    image_files = sorted(IMAGE_DIR.glob("*_CBCT_Image.nii.gz"))
    case_ids = [get_case_id(path) for path in image_files]

    groups: dict[str, list[str]] = defaultdict(list)

    for case_id in case_ids:
        lesion_code = case_id.split("_")[0]
        groups[lesion_code].append(case_id)

    rows: list[dict[str, str]] = []

    for lesion_code, lesion_cases in sorted(groups.items()):
        train, val, test = split_group(lesion_cases)

        for case_id in train:
            rows.append(
                {
                    "case_id": case_id,
                    "lesion_code": lesion_code,
                    "split": "train",
                }
            )

        for case_id in val:
            rows.append(
                {
                    "case_id": case_id,
                    "lesion_code": lesion_code,
                    "split": "val",
                }
            )

        for case_id in test:
            rows.append(
                {
                    "case_id": case_id,
                    "lesion_code": lesion_code,
                    "split": "test",
                }
            )

    rows.sort(key=lambda row: (row["split"], row["lesion_code"], row["case_id"]))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case_id", "lesion_code", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    split_counts: dict[str, int] = defaultdict(int)
    lesion_split_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        split_counts[row["split"]] += 1
        lesion_split_counts[(row["lesion_code"], row["split"])] += 1

    print("=" * 50)
    print("OralVision AI — Dataset Splits")
    print("=" * 50)

    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Total cases: {len(rows)}")

    print("\nOverall split counts:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {split_counts[split]}")

    print("\nCounts by lesion and split:")
    for lesion_code in sorted(groups):
        print(f"  {lesion_code}:")
        for split in ("train", "val", "test"):
            count = lesion_split_counts[(lesion_code, split)]
            print(f"    {split}: {count}")


if __name__ == "__main__":
    main()
    