"""Smoke-test the first-stage OralVisionAI coarse lesion localizer.

RESEARCH USE ONLY — NOT FOR CLINICAL USE.
"""

from __future__ import annotations

from itertools import cycle
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.losses import DiceCELoss

from src.data.cached_dataset import build_data_list
from src.data.localization_transforms import (
    LOCALIZATION_SIZE,
    get_localization_transforms,
)
from src.models.localizer3d import create_localizer3d


NUM_CASES = 5
NUM_STEPS = 10
CHECKPOINT_PATH = Path("checkpoints/localizer_v1/smoke_test.pt")
EXPECTED_SHAPE = (1, 1, *LOCALIZATION_SIZE)


def _validate_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
    """Check shapes, finite values, and binary labels before optimization."""
    if tuple(images.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected image shape: {tuple(images.shape)}; "
            f"expected {EXPECTED_SHAPE}"
        )
    if tuple(labels.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected label shape: {tuple(labels.shape)}; "
            f"expected {EXPECTED_SHAPE}"
        )
    if not bool(torch.isfinite(images).all().item()):
        raise ValueError("Images contain non-finite values.")
    if not bool(torch.isfinite(labels).all().item()):
        raise ValueError("Labels contain non-finite values.")
    if not bool(torch.logical_or(labels == 0, labels == 1).all().item()):
        unique_values = torch.unique(labels).detach().cpu().tolist()
        raise ValueError(f"Labels are not binary: {unique_values}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_data_list(
        split="train",
        case_indices=range(NUM_CASES),
    )
    dataset = Dataset(
        data=data,
        transform=get_localization_transforms(),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_localizer3d().to(device)
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("=" * 72)
    print("OralVision AI — LOCALIZER SMOKE TEST")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Training cases: {len(dataset)}")
    print(f"Optimization steps: {NUM_STEPS}")
    print(f"Expected input shape: {EXPECTED_SHAPE}")

    model.train()
    batch_iterator = cycle(loader)
    latest_loss = float("nan")

    for step in range(1, NUM_STEPS + 1):
        batch = next(batch_iterator)
        images = batch["image"]
        labels = batch["label"]
        _validate_batch(images, labels)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            if tuple(logits.shape) != EXPECTED_SHAPE:
                raise ValueError(
                    f"Unexpected output shape: {tuple(logits.shape)}; "
                    f"expected {EXPECTED_SHAPE}"
                )
            loss = loss_function(logits, labels)

        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        latest_loss = float(loss.item())

        print(
            f"Step {step:02d}/{NUM_STEPS} | "
            f"Input={tuple(images.shape)} | "
            f"Output={tuple(logits.shape)} | "
            f"Loss={latest_loss:.4f}"
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": "localizer_v1_smoke_test",
            "research_use_only": True,
            "steps_completed": NUM_STEPS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "latest_loss": latest_loss,
            "localization_size": LOCALIZATION_SIZE,
            "training_cases": [item["case_id"] for item in data],
        },
        CHECKPOINT_PATH,
    )

    print("\nSmoke test completed successfully.")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory allocated: {peak_memory_gb:.2f} GB")
    print("RESEARCH USE ONLY — NOT FOR CLINICAL USE.")


if __name__ == "__main__":
    main()
