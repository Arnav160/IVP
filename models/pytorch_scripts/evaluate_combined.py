
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "models" / "unet_resnet_combined"))

from combined_dataset import CombinedDataset
from model import create_model


def calculate_metrics(predictions, targets, smooth=1e-7):
    predictions = predictions.bool()
    targets = targets.bool()

    intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
    pred_pixels = predictions.sum(dim=(1, 2, 3)).float()
    target_pixels = targets.sum(dim=(1, 2, 3)).float()

    dice = (2 * intersection + smooth) / (
        pred_pixels + target_pixels + smooth
    )
    iou = (intersection + smooth) / (
        pred_pixels + target_pixels - intersection + smooth
    )

    return dice, iou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    checkpoint_path = (
        PROJECT_DIR / "checkpoints"
        / "unet_resnet34_rgb_canny_depth_best.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = create_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded best checkpoint from epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["best_val_loss"])

    datasets = ("COD10K_CAM", "CAMO", "NC4K")
    results = []

    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name}...")

        dataset = CombinedDataset(
            project_dir=PROJECT_DIR,
            datasets=(dataset_name,),
            split="test",
            image_size=352,
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        dice_sum = 0.0
        iou_sum = 0.0
        image_count = 0

        global_intersection = 0
        global_pred_pixels = 0
        global_target_pixels = 0

        with torch.no_grad():
            for inputs, masks in tqdm(loader, desc=dataset_name):
                inputs = inputs.to(device)
                masks = masks.to(device)

                logits = model(inputs)
                predictions = torch.sigmoid(logits) >= 0.5
                targets = masks >= 0.5

                dice, iou = calculate_metrics(predictions, targets)

                batch_count = inputs.size(0)
                dice_sum += dice.sum().item()
                iou_sum += iou.sum().item()
                image_count += batch_count

                global_intersection += (
                    (predictions & targets).sum().item()
                )
                global_pred_pixels += predictions.sum().item()
                global_target_pixels += targets.sum().item()

        mean_dice = dice_sum / image_count
        mean_iou = iou_sum / image_count

        global_dice = (
            2 * global_intersection
            / max(global_pred_pixels + global_target_pixels, 1)
        )
        global_iou = (
            global_intersection
            / max(
                global_pred_pixels
                + global_target_pixels
                - global_intersection,
                1,
            )
        )

        results.append(
            (
                dataset_name,
                image_count,
                mean_dice,
                mean_iou,
                global_dice,
                global_iou,
            )
        )

    print("\n" + "=" * 85)
    print(
        f"{'Dataset':<15} {'Images':>7} {'Mean Dice':>11} "
        f"{'Mean IoU':>11} {'Global Dice':>13} {'Global IoU':>12}"
    )
    print("-" * 85)

    for row in results:
        name, count, md, mi, gd, gi = row
        print(
            f"{name:<15} {count:>7} {md:>11.6f} "
            f"{mi:>11.6f} {gd:>13.6f} {gi:>12.6f}"
        )

    print("=" * 85)


if __name__ == "__main__":
    main()
