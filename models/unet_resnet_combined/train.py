
import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Resolve project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
PYTORCH_DIR = PROJECT_DIR / "pytorch"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Import the combined dataset and this model
sys.path.insert(0, str(PYTORCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combined_dataset import CombinedDataset
from model import create_model


def dice_loss(logits, targets, smooth=1e-6):
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    image_size = 352
    batch_size = 4
    epochs = 20
    learning_rate = 1e-5

    datasets = ("COD10K_CAM", "CAMO", "NC4K")

    print("\nLoading training dataset...")
    train_dataset = CombinedDataset(
        project_dir=PROJECT_DIR,
        datasets=datasets,
        split="train",
        image_size=image_size,
    )

    print("\nLoading validation dataset...")
    val_dataset = CombinedDataset(
        project_dir=PROJECT_DIR,
        datasets=datasets,
        split="val",
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("\nCreating 5-channel U-Net...")
    model = create_model().to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    checkpoint_path = (
        CHECKPOINT_DIR / "unet_resnet34_rgb_canny_depth_best.pth"
    )
    csv_path = (
        CHECKPOINT_DIR / "unet_resnet34_rgb_canny_depth_training.csv"
    )

    best_val_loss = float("inf")

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
        ])

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss_sum = 0.0
            train_samples = 0

            train_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            )

            for inputs, masks in train_bar:
                inputs = inputs.to(
                    device, non_blocking=True
                )
                masks = masks.to(
                    device, non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)

                logits = model(inputs)
                loss = (
                    bce_loss(logits, masks)
                    + dice_loss(logits, masks)
                )

                loss.backward()
                optimizer.step()

                batch_count = inputs.size(0)
                train_loss_sum += loss.item() * batch_count
                train_samples += batch_count

                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = train_loss_sum / train_samples

            # Validation
            model.eval()
            val_loss_sum = 0.0
            val_samples = 0

            with torch.no_grad():
                val_bar = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [Val]",
                )

                for inputs, masks in val_bar:
                    inputs = inputs.to(
                        device, non_blocking=True
                    )
                    masks = masks.to(
                        device, non_blocking=True
                    )

                    logits = model(inputs)
                    loss = (
                        bce_loss(logits, masks)
                        + dice_loss(logits, masks)
                    )

                    batch_count = inputs.size(0)
                    val_loss_sum += loss.item() * batch_count
                    val_samples += batch_count

                    val_bar.set_postfix(loss=f"{loss.item():.4f}")

            val_loss = val_loss_sum / val_samples
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f}"
            )

            writer.writerow([
                epoch + 1,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                current_lr,
            ])
            csv_file.flush()

            # Save only the best combined-model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "image_size": image_size,
                        "input_channels": 5,
                        "datasets": list(datasets),
                    },
                    checkpoint_path,
                )

                print("Saved new best checkpoint:", checkpoint_path)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print("Best checkpoint:", checkpoint_path)
    print("Training log:", csv_path)


if __name__ == "__main__":
    main()
