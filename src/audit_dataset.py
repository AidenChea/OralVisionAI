from collections import Counter
from pathlib import Path


IMAGE_DIR = Path("data/DOLCHID/cbct_image")
LABEL_DIR = Path("data/DOLCHID/cbct_label")


def case_id_from_image(path: Path) -> str:
    return path.name.replace("_CBCT_Image.nii.gz", "")


def case_id_from_label(path: Path) -> str:
    return path.name.replace("_CBCT_Label.nii.gz", "")


def main() -> None:
    image_files = sorted(IMAGE_DIR.glob("*_CBCT_Image.nii.gz"))
    label_files = sorted(LABEL_DIR.glob("*_CBCT_Label.nii.gz"))

    image_ids = {case_id_from_image(path) for path in image_files}
    label_ids = {case_id_from_label(path) for path in label_files}

    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)
    matched_ids = sorted(image_ids & label_ids)

    lesion_counts = Counter(
        case_id.split("_")[0] for case_id in matched_ids
    )

    print("=" * 50)
    print("OralVision AI — Dataset Audit")
    print("=" * 50)

    print(f"CBCT images: {len(image_files)}")
    print(f"CBCT labels: {len(label_files)}")
    print(f"Matched cases: {len(matched_ids)}")

    print("\nCases by lesion code:")
    for lesion_code, count in sorted(lesion_counts.items()):
        print(f"  {lesion_code}: {count}")

    print("\nMissing labels:")
    if missing_labels:
        for case_id in missing_labels:
            print(f"  {case_id}")
    else:
        print("  None")

    print("\nMissing images:")
    if missing_images:
        for case_id in missing_images:
            print(f"  {case_id}")
    else:
        print("  None")

    if matched_ids:
        print("\nFirst five matched cases:")
        for case_id in matched_ids[:5]:
            print(f"  {case_id}")


if __name__ == "__main__":
    main()
    