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

def evaluate_dataset(dataset_name):
    """Evaluate model on a specific dataset and return metrics."""

    print(f"\nLoading {dataset_name} test dataloader...")

    _, _, test_loader = get_dataloaders(dataset_name=dataset_name)

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )
    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )

    # Create model
    print("\nCreating model...")

    model = create_model()

    model = model.to(DEVICE)

    # Load checkpoint
    print("\nLoading best checkpoint...")

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    print("Checkpoint loaded successfully.")

    # Evaluation mode
    model.eval()

    total_dice = 0.0
    total_iou = 0.0

    # Test loop
    print(f"\nEvaluating {dataset_name} test dataset...\n")

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

    # Average results
    average_dice = (
        total_dice / len(test_loader)
    )

    average_iou = (
        total_iou / len(test_loader)
    )

    return average_dice, average_iou, len(test_loader.dataset), len(test_loader)


def main():
    print("=" * 70)
    print("CROSS-DATASET U-NET RESNET-34 EVALUATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"\nCheckpoint:")
    print(f"{CHECKPOINT_PATH}")

    # Evaluate on all three datasets
    datasets = ["COD10K_CAM", "CAMO", "NC4K"]
    results = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"{dataset_name} TEST")
        print(f"{'='*70}")

        try:
            avg_dice, avg_iou, num_samples, num_batches = evaluate_dataset(dataset_name)
            results[dataset_name] = (avg_dice, avg_iou, num_samples, num_batches)

            print(f"\nAverage Dice Score: {avg_dice:.6f}")
            print(f"Average IoU Score:  {avg_iou:.6f}")

        except Exception as e:
            print(f"\nERROR evaluating {dataset_name}: {e}")
            results[dataset_name] = (0.0, 0.0, 0, 0)

    # Final cross-dataset results
    print(f"\n{'='*70}")
    print("FINAL CROSS-DATASET RESULTS")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Dice Score':<15} {'IoU Score':<15}")
    print(f"{'-'*45}")

    for dataset_name in datasets:
        if dataset_name in results:
            avg_dice, avg_iou, _, _ = results[dataset_name]
            print(f"{dataset_name:<15} {avg_dice:<15.6f} {avg_iou:<15.6f}")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETED")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()