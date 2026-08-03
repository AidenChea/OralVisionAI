from dataset import OralVisionDataset


def main() -> None:
    dataset = OralVisionDataset(split="train")

    print("=" * 45)
    print("OralVision AI — Dataset Test")
    print("=" * 45)

    print(f"Training cases: {len(dataset)}")

    sample = dataset[0]

    print(f"Case ID: {sample['case_id']}")
    print(f"Lesion code: {sample['lesion_code']}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label shape: {sample['label'].shape}")
    print(f"Voxel spacing: {sample['spacing']}")
    print(
        "Tumor voxels:",
        int((sample["label"] > 0).sum()),
    )


if __name__ == "__main__":
    main()
    