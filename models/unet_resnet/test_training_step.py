from pathlib import Path
import sys

import torch
import torch.nn as nn


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = CURRENT_DIR.parent.parent

PYTORCH_DIR = PROJECT_DIR / "pytorch"

sys.path.append(str(PYTORCH_DIR))


# ==================================================
# IMPORTS
# ==================================================

from dataloader import get_dataloaders
from model import create_model


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("U-NET RESNET-34 TRAINING STEP TEST")
    print("=" * 60)

    # ----------------------------------------------
    # DEVICE
    # ----------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}")

    # ----------------------------------------------
    # MODEL
    # ----------------------------------------------

    print("\nCreating model...")

    model = create_model().to(device)

    # Training mode
    model.train()

    print("Model created successfully.")

    # ----------------------------------------------
    # LOSS FUNCTION
    # ----------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    print("\nLoss function:")
    print("BCEWithLogitsLoss")

    # ----------------------------------------------
    # OPTIMIZER
    # ----------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    print("\nOptimizer:")
    print("Adam")

    print("Learning rate: 0.0001")

    # ----------------------------------------------
    # LOAD ONE BATCH
    # ----------------------------------------------

    print("\nLoading training batch...")

    train_loader, _, _ = get_dataloaders()

    images, masks = next(iter(train_loader))

    images = images.to(device)
    masks = masks.to(device)

    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape:  {masks.shape}")

    # ----------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------

    print("\nRunning forward pass...")

    outputs = model(images)

    print(
        f"Model output shape: {outputs.shape}"
    )

    # ----------------------------------------------
    # CALCULATE LOSS
    # ----------------------------------------------

    loss = criterion(
        outputs,
        masks
    )

    print(
        f"\nLoss before update: "
        f"{loss.item():.6f}"
    )

    # ----------------------------------------------
    # BACKWARD PASS
    # ----------------------------------------------

    print("\nRunning backward pass...")

    optimizer.zero_grad()

    loss.backward()

    print("Backward pass successful.")

    # ----------------------------------------------
    # OPTIMIZER UPDATE
    # ----------------------------------------------

    optimizer.step()

    print("Optimizer update successful.")

    # ----------------------------------------------
    # VERIFY SECOND FORWARD PASS
    # ----------------------------------------------

    print("\nRunning second forward pass...")

    outputs_after = model(images)

    loss_after = criterion(
        outputs_after,
        masks
    )

    print(
        f"\nLoss after update: "
        f"{loss_after.item():.6f}"
    )

    print("\n" + "=" * 60)
    print("TRAINING STEP TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()