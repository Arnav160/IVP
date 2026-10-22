from pathlib import Path
import sys

import torch


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent.parent

PYTORCH_DIR = PROJECT_DIR / "pytorch"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

sys.path.append(str(PYTORCH_DIR))


# ============================================================
# IMPORTS
# ============================================================

from dataloader import get_dataloaders
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR /
    "unet_resnet34_all_datasets_best.pth"
)

DATASETS = [
    "COD10K_CAM",
    "CAMO",
    "NC4K"
]


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(outputs, masks, threshold=0.5, smooth=1.0):

    probabilities = torch.sigmoid(outputs)

    predictions = (
        probabilities > threshold
    ).float()

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


# ============================================================
# IOU SCORE
# ============================================================

def iou_score(outputs, masks, threshold=0.5, smooth=1.0):

    probabilities = torch.sigmoid(outputs)

    predictions = (
        probabilities > threshold
    ).float()

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


# ============================================================
# EVALUATE ONE DATASET
# ============================================================

def evaluate_dataset(
    model,
    dataset_name
):

    print("\n" + "=" * 70)
    print(f"{dataset_name} TEST")
    print("=" * 70)

    print(
        f"\nLoading {dataset_name} test dataloader..."
    )

    _, _, test_loader = get_dataloaders(
        dataset_name=dataset_name
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )

    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    model.eval()

    print(
        f"\nEvaluating {dataset_name}..."
    )

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

            outputs = model(images)

            dice = dice_score(
                outputs,
                masks
            )

            iou = iou_score(
                outputs,
                masks
            )

            total_dice += dice
            total_iou += iou

            total_batches += 1

            if (
                batch_index + 1
            ) % 50 == 0:

                print(
                    f"Processed "
                    f"{batch_index + 1}/"
                    f"{len(test_loader)} batches"
                )

    average_dice = (
        total_dice / total_batches
    )

    average_iou = (
        total_iou / total_batches
    )

    print(
        f"\nAverage Dice Score: "
        f"{average_dice:.6f}"
    )

    print(
        f"Average IoU Score:  "
        f"{average_iou:.6f}"
    )

    return average_dice, average_iou


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COMBINED U-NET RESNET-34 TEST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nChecking checkpoint..."
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    if not CHECKPOINT_PATH.exists():

        print(
            "\nERROR: Combined checkpoint "
            "was not found!"
        )

        sys.exit(1)

    print(
        "Checkpoint found."
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print(
        "\nCreating U-Net ResNet-34 model..."
    )

    model = create_model().to(
        DEVICE
    )

    print(
        "Model created successfully."
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nLoading combined checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # EVALUATE ALL DATASETS
    # --------------------------------------------------------

    results = {}

    for dataset_name in DATASETS:

        dice, iou = evaluate_dataset(
            model,
            dataset_name
        )

        results[dataset_name] = {
            "dice": dice,
            "iou": iou
        }

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL CROSS-DATASET RESULTS")
    print("=" * 70)

    print(
        f"{'Dataset':<20}"
        f"{'Dice Score':<18}"
        f"{'IoU Score':<18}"
    )

    print("-" * 56)

    for dataset_name in DATASETS:

        dice = results[
            dataset_name
        ]["dice"]

        iou = results[
            dataset_name
        ]["iou"]

        print(
            f"{dataset_name:<20}"
            f"{dice:<18.6f}"
            f"{iou:<18.6f}"
        )

    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()