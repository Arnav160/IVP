from pathlib import Path
import sys

import torch


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = CURRENT_DIR.parent.parent

PYTORCH_DIR = (
    PROJECT_DIR
    / "pytorch"
)

sys.path.append(str(PYTORCH_DIR))
sys.path.append(str(CURRENT_DIR))


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
    print("U-NET RESNET-34 MODEL TEST")
    print("=" * 60)

    # ----------------------------------------------
    # DEVICE
    # ----------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(f"\nUsing device: {device}")

    if device.type == "cuda":

        print(
            "GPU name: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ----------------------------------------------
    # CREATE MODEL
    # ----------------------------------------------

    print("\nCreating U-Net model...")

    model = create_model()

    model = model.to(device)

    print("Model created successfully.")

    # ----------------------------------------------
    # LOAD DATA
    # ----------------------------------------------

    print("\nLoading dataset batch...")

    train_loader, _, _ = get_dataloaders()

    images, masks = next(iter(train_loader))

    images = images.to(device)
    masks = masks.to(device)

    print(
        f"Input image shape: {images.shape}"
    )

    print(
        f"Ground truth mask shape: {masks.shape}"
    )

    # ----------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------

    print("\nRunning forward pass...")

    with torch.no_grad():

        outputs = model(images)

    print("\nForward pass successful.")

    print(
        f"Output shape: {outputs.shape}"
    )

    print(
        f"Output dtype: {outputs.dtype}"
    )

    print(
        f"Output device: {outputs.device}"
    )

    # ----------------------------------------------
    # VERIFY OUTPUT
    # ----------------------------------------------

    expected_shape = masks.shape

    print("\nExpected output shape:")

    print(expected_shape)

    print("\n" + "=" * 60)

    if outputs.shape == expected_shape:

        print(
            "U-NET RESNET-34 TEST PASSED"
        )

    else:

        print(
            "WARNING: OUTPUT SHAPE DOES NOT MATCH MASK"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()