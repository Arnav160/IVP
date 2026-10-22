
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "pytorch"))

from depth_dataset import RGBDepthDataset
from model import create_model

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "unet_resnet34_depth_best.pth"
LOG_PATH = CHECKPOINT_DIR / "unet_resnet34_depth_training.csv"

# --------------------------------------------------
# Training configuration
# --------------------------------------------------
IMAGE_SIZE = 352
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-5
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# BCE + Dice loss
# --------------------------------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probabilities = torch.sigmoid(logits)
        smooth = 1.0

        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = (
            probabilities.sum(dim=(1, 2, 3))
            + targets.sum(dim=(1, 2, 3))
        )

        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        dice_loss = 1.0 - dice.mean()

        return bce_loss + dice_loss


# --------------------------------------------------
# One training epoch
# --------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc="Training", leave=False)

    for inputs, masks in progress:
        inputs = inputs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples


# --------------------------------------------------
# Validation
# --------------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc="Validation", leave=False)

    for inputs, masks in progress:
        inputs = inputs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, masks)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples


# --------------------------------------------------
# Main training function
# --------------------------------------------------
def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RGB + DEPTH U-NET TRAINING")
    print("=" * 60)
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)

    train_dataset = RGBDepthDataset(
        project_dir=PROJECT_DIR,
        split="train",
        image_size=IMAGE_SIZE,
    )

    val_dataset = RGBDepthDataset(
        project_dir=PROJECT_DIR,
        split="val",
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = create_model().to(DEVICE)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")

    # Start a fresh log for this training run.
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "best_val_loss"])

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion
            )
            val_loss = validate(
                model, val_loader, criterion
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "image_size": IMAGE_SIZE,
                        "input_channels": 4,
                        "model_name": "Unet-ResNet34-RGB-Depth",
                    },
                    CHECKPOINT_PATH,
                )

            writer.writerow(
                [epoch, train_loss, val_loss, best_val_loss]
            )
            csv_file.flush()

            status = "  <-- BEST CHECKPOINT SAVED" if is_best else ""
            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Best Val Loss: {best_val_loss:.5f}"
                f"{status}"
            )

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")
    print(f"Training log: {LOG_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
