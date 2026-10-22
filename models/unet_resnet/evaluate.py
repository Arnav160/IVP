from pathlib import Path
import sys

import torch


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = CURRENT_DIR.parent.parent

PYTORCH_DIR = PROJECT_DIR / "pytorch"

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "unet_resnet34_best.pth"
)

sys.path.append(str(PYTORCH_DIR))


# ==================================================
# IMPORTS
# ==================================================

from dataloader import get_dataloaders
from model import create_model


# ==================================================
# DEVICE
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# METRICS
# ==================================================

def calculate_dice(
    predictions,
    masks,
    smooth=1e-7
):

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    masks = masks.view(
        masks.size(0),
        -1
    )

    intersection = (
        predictions * masks
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum(dim=1)
        + masks.sum(dim=1)
        + smooth
    )

    return dice.mean().item()


def calculate_iou(
    predictions,
    masks,
    smooth=1e-7
):

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    masks = masks.view(
        masks.size(0),
        -1
    )

    intersection = (
        predictions * masks
    ).sum(dim=1)

    union = (
        predictions.sum(dim=1)
        + masks.sum(dim=1)
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean().item()


# ==================================================
# EVALUATION
# ==================================================

def main():

    print("=" * 60)
    print("U-NET RESNET-34 TEST EVALUATION")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    # ----------------------------------------------
    # LOAD TEST DATA
    # ----------------------------------------------

    print("\nLoading test dataloader...")

    _, _, test_loader = get_dataloaders()

    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )

    # ----------------------------------------------
    # CREATE MODEL
    # ----------------------------------------------

    print("\nCreating model...")

    model = create_model()

    model = model.to(DEVICE)

    # ----------------------------------------------
    # LOAD CHECKPOINT
    # ----------------------------------------------

    print("\nLoading best checkpoint...")

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    print("Checkpoint loaded successfully.")

    # ----------------------------------------------
    # EVALUATION MODE
    # ----------------------------------------------

    model.eval()

    total_dice = 0.0
    total_iou = 0.0

    # ----------------------------------------------
    # TEST LOOP
    # ----------------------------------------------

    print("\nEvaluating test dataset...\n")

    with torch.no_grad():

        for batch_index, (
            images,
            masks
        ) in enumerate(test_loader):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            # Model output (logits)
            outputs = model(images)

            # Convert logits to probabilities
            probabilities = torch.sigmoid(
                outputs
            )

            # Convert probabilities to binary masks
            predictions = (
                probabilities >= 0.5
            ).float()

            # Calculate metrics
            dice = calculate_dice(
                predictions,
                masks
            )

            iou = calculate_iou(
                predictions,
                masks
            )

            total_dice += dice
            total_iou += iou

            # Progress
            if (
                (batch_index + 1) % 50 == 0
            ):

                print(
                    f"Processed "
                    f"{batch_index + 1}/"
                    f"{len(test_loader)} batches"
                )

    # ----------------------------------------------
    # AVERAGE RESULTS
    # ----------------------------------------------

    average_dice = (
        total_dice / len(test_loader)
    )

    average_iou = (
        total_iou / len(test_loader)
    )

    # ==============================================
    # RESULTS
    # ==============================================

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"\nAverage Dice Score: "
        f"{average_dice:.6f}"
    )

    print(
        f"Average IoU Score:  "
        f"{average_iou:.6f}"
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()