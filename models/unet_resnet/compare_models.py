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

DATASETS = [
    "COD10K_CAM",
    "CAMO",
    "NC4K"
]

CHECKPOINTS = {
    "Original COD10K Model":
        CHECKPOINT_DIR / "unet_resnet34_best.pth",

    "Combined Model":
        CHECKPOINT_DIR / "unet_resnet34_all_datasets_best.pth"
}


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
# EVALUATE ONE MODEL ON ONE DATASET
# ============================================================

def evaluate_model_on_dataset(
    model,
    dataset_name
):

    print(
        f"\n  Evaluating on {dataset_name}..."
    )

    _, _, test_loader = get_dataloaders(
        dataset_name=dataset_name
    )

    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    model.eval()

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
            ) % 100 == 0:

                print(
                    f"    Processed "
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
        f"    Dice: {average_dice:.6f}"
    )

    print(
        f"    IoU:  {average_iou:.6f}"
    )

    return average_dice, average_iou


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("U-NET RESNET-34 MODEL COMPARISON")
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
    # CHECKPOINTS
    # --------------------------------------------------------

    print(
        "\nChecking checkpoints..."
    )

    for model_name, checkpoint_path in CHECKPOINTS.items():

        print(
            f"\n{model_name}:"
        )

        print(
            f"  {checkpoint_path}"
        )

        if not checkpoint_path.exists():

            print(
                "  ERROR: Checkpoint not found!"
            )

            sys.exit(1)

        print(
            "  Checkpoint found."
        )

    # --------------------------------------------------------
    # RESULTS STORAGE
    # --------------------------------------------------------

    all_results = {}

    # --------------------------------------------------------
    # EVALUATE EACH MODEL
    # --------------------------------------------------------

    for model_name, checkpoint_path in CHECKPOINTS.items():

        print("\n")
        print("=" * 70)
        print(model_name.upper())
        print("=" * 70)

        # Create model
        print(
            "\nCreating U-Net ResNet-34 model..."
        )

        model = create_model().to(
            DEVICE
        )

        # Load checkpoint
        print(
            "Loading checkpoint..."
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint
        )

        print(
            "Checkpoint loaded successfully."
        )

        # Results for this model
        all_results[model_name] = {}

        # Evaluate all datasets
        for dataset_name in DATASETS:

            dice, iou = (
                evaluate_model_on_dataset(
                    model,
                    dataset_name
                )
            )

            all_results[
                model_name
            ][dataset_name] = {
                "dice": dice,
                "iou": iou
            }

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n\n")
    print("=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(
        f"\n{'Model':<25}"
        f"{'Dataset':<18}"
        f"{'Dice':<15}"
        f"{'IoU':<15}"
    )

    print("-" * 73)

    for model_name in CHECKPOINTS:

        for dataset_name in DATASETS:

            dice = all_results[
                model_name
            ][dataset_name]["dice"]

            iou = all_results[
                model_name
            ][dataset_name]["iou"]

            print(
                f"{model_name:<25}"
                f"{dataset_name:<18}"
                f"{dice:<15.6f}"
                f"{iou:<15.6f}"
            )

        print("-" * 73)

    # ========================================================
    # IMPROVEMENT SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("IMPROVEMENT: COMBINED MODEL VS ORIGINAL")
    print("=" * 80)

    original_name = "Original COD10K Model"
    combined_name = "Combined Model"

    for dataset_name in DATASETS:

        original_dice = (
            all_results[
                original_name
            ][dataset_name]["dice"]
        )

        combined_dice = (
            all_results[
                combined_name
            ][dataset_name]["dice"]
        )

        original_iou = (
            all_results[
                original_name
            ][dataset_name]["iou"]
        )

        combined_iou = (
            all_results[
                combined_name
            ][dataset_name]["iou"]
        )

        dice_change = (
            combined_dice - original_dice
        )

        iou_change = (
            combined_iou - original_iou
        )

        print(
            f"\n{dataset_name}"
        )

        print(
            f"  Dice change: "
            f"{dice_change:+.6f}"
        )

        print(
            f"  IoU change:  "
            f"{iou_change:+.6f}"
        )

    print("\n")
    print("=" * 80)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()