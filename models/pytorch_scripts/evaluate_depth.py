
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------------
# Paths and imports
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "models" / "unet_resnet_depth"))

from depth_dataset import RGBDepthDataset
from model import create_model

CHECKPOINT_PATH = (
    PROJECT_DIR / "checkpoints" / "unet_resnet34_depth_best.pth"
)

DATASETS = ["COD10K_CAM", "CAMO", "NC4K"]
IMAGE_SIZE = 352
BATCH_SIZE = 4
NUM_WORKERS = 0
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_dataset(model, dataset_name):
    dataset = RGBDepthDataset(
        project_dir=PROJECT_DIR,
        datasets=(dataset_name,),
        split="test",
        image_size=IMAGE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    total_intersection = 0
    total_predicted = 0
    total_ground_truth = 0
    total_union = 0
    total_images = 0
    image_dice_sum = 0.0
    image_iou_sum = 0.0
    smooth = 1e-7

    model.eval()

    for inputs, masks in tqdm(loader, desc=f"Testing {dataset_name}"):
        inputs = inputs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        logits = model(inputs)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= THRESHOLD).float()

        # Calculate per-image scores
        for pred, target in zip(predictions, masks):
            pred = pred.bool()
            target = target.bool()

            intersection = (pred & target).sum().item()
            predicted = pred.sum().item()
            ground_truth = target.sum().item()
            union = (pred | target).sum().item()

            dice = (2 * intersection + smooth) / (
                predicted + ground_truth + smooth
            )
            iou = (intersection + smooth) / (union + smooth)

            image_dice_sum += dice
            image_iou_sum += iou
            total_images += 1

            total_intersection += intersection
            total_predicted += predicted
            total_ground_truth += ground_truth
            total_union += union

    # Mean per-image metrics, matching common segmentation reporting
    mean_dice = image_dice_sum / total_images
    mean_iou = image_iou_sum / total_images

    # Also report global pixel-level metrics
    global_dice = (2 * total_intersection + smooth) / (
        total_predicted + total_ground_truth + smooth
    )
    global_iou = (total_intersection + smooth) / (
        total_union + smooth
    )

    return mean_dice, mean_iou, global_dice, global_iou, total_images


def main():
    print("=" * 65)
    print("RGB + DEPTH U-NET TEST EVALUATION")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = create_model().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Best training epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.5f}")

    results = []

    for dataset_name in DATASETS:
        mean_dice, mean_iou, global_dice, global_iou, count = (
            evaluate_dataset(model, dataset_name)
        )

        results.append(
            (
                dataset_name,
                count,
                mean_dice,
                mean_iou,
                global_dice,
                global_iou,
            )
        )

    print("\n" + "=" * 90)
    print("RGB + DEPTH TEST RESULTS")
    print("=" * 90)
    print(
        f"{'Dataset':<15} {'Images':>7} {'Mean Dice':>11} "
        f"{'Mean IoU':>10} {'Global Dice':>12} {'Global IoU':>11}"
    )
    print("-" * 90)

    for name, count, dice, iou, global_dice, global_iou in results:
        print(
            f"{name:<15} {count:>7} {dice:>11.6f} "
            f"{iou:>10.6f} {global_dice:>12.6f} "
            f"{global_iou:>11.6f}"
        )

    # Save results to CSV
    output_path = (
        PROJECT_DIR / "checkpoints" / "unet_resnet34_depth_test_results.csv"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(
            "dataset,images,mean_dice,mean_iou,global_dice,global_iou\n"
        )
        for row in results:
            file.write(
                f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},"
                f"{row[4]:.6f},{row[5]:.6f}\n"
            )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
