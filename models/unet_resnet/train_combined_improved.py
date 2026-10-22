from pathlib import Path
import sys

import torch
import torch.nn as nn


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

from dataloader import get_combined_dataloaders
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = [
    "COD10K_CAM",
    "CAMO",
    "NC4K"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Maximum number of epochs
EPOCHS = 20

# Starting learning rate
LEARNING_RATE = 1e-5

# Reduce LR when validation loss stops improving
LR_PATIENCE = 2

# Amount to reduce learning rate
LR_FACTOR = 0.5

# Stop training after this many epochs without improvement
EARLY_STOPPING_PATIENCE = 5

# Original COD10K checkpoint
STARTING_CHECKPOINT = (
    CHECKPOINT_DIR /
    "unet_resnet34_best.pth"
)

# NEW checkpoint - does not overwrite previous models
OUTPUT_CHECKPOINT = (
    CHECKPOINT_DIR /
    "unet_resnet34_improved_best.pth"
)


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(outputs, masks, smooth=1.0):

    probabilities = torch.sigmoid(outputs)

    probabilities = probabilities.view(
        probabilities.size(0),
        -1
    )

    masks = masks.view(
        masks.size(0),
        -1
    )

    intersection = (
        probabilities * masks
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        probabilities.sum(dim=1)
        + masks.sum(dim=1)
        + smooth
    )

    return 1.0 - dice.mean()


# ============================================================
# COMBINED LOSS
# ============================================================

def combined_loss(
    outputs,
    masks,
    bce_criterion
):

    bce = bce_criterion(
        outputs,
        masks
    )

    dice = dice_loss(
        outputs,
        masks
    )

    return bce + dice


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    bce_criterion
):

    model.train()

    total_loss = 0.0

    for batch_index, (images, masks) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(
            outputs,
            masks,
            bce_criterion
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 500 == 0:

            print(
                f"  Training batch "
                f"{batch_index + 1}/"
                f"{len(loader)}"
            )

    return total_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    bce_criterion
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks,
                bce_criterion
            )

            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("U-NET RESNET-34 IMPROVED COMBINED TRAINING")
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
    # CHECK STARTING CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nChecking starting checkpoint..."
    )

    print(
        f"Starting checkpoint:"
        f"\n{STARTING_CHECKPOINT}"
    )

    if not STARTING_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"\nStarting checkpoint not found:\n"
            f"{STARTING_CHECKPOINT}"
        )

    print(
        "Starting checkpoint found."
    )

    # --------------------------------------------------------
    # OUTPUT CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nOutput checkpoint:"
        f"\n{OUTPUT_CHECKPOINT}"
    )

    if OUTPUT_CHECKPOINT.exists():

        print(
            "\nWARNING:"
            "\nImproved checkpoint already exists."
        )

        print(
            "This script will overwrite it if a"
            " better model is found."
        )

    # --------------------------------------------------------
    # LOAD COMBINED DATASETS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LOADING COMBINED DATASETS")
    print("=" * 70)

    print("\nDatasets:")

    for dataset_name in DATASETS:

        print(
            f"  - {dataset_name}"
        )

    print(
        "\nCreating combined dataloaders..."
    )

    (
        train_loader,
        val_loader,
        _
    ) = get_combined_dataloaders(
        DATASETS
    )

    print(
        f"\nCombined training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Combined validation samples: "
        f"{len(val_loader.dataset)}"
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = create_model().to(
        DEVICE
    )

    print(
        "Model created successfully."
    )

    # --------------------------------------------------------
    # LOAD ORIGINAL CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nLoading original COD10K checkpoint..."
    )

    checkpoint = torch.load(
        STARTING_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    print(
        "Original checkpoint loaded successfully."
    )

    print(
        "\nTraining starts from:"
    )

    print(
        "  unet_resnet34_best.pth"
    )

    print(
        "\nThe previous combined checkpoints"
        "\nwill NOT be used."
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    bce_criterion = (
        nn.BCEWithLogitsLoss()
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # TRAINING INFORMATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STARTING IMPROVED TRAINING")
    print("=" * 70)

    print(
        f"\nMaximum epochs: {EPOCHS}"
    )

    print(
        f"Initial learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"LR scheduler patience: "
        f"{LR_PATIENCE}"
    )

    print(
        f"LR reduction factor: "
        f"{LR_FACTOR}"
    )

    print(
        f"Early stopping patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        print("\n")
        print("-" * 70)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print("-" * 70)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            bce_criterion
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        val_loss = validate(
            model,
            val_loader,
            bce_criterion
        )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"\nTrain Loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.6f}"
        )

        # ----------------------------------------------------
        # LEARNING RATE SCHEDULER
        # ----------------------------------------------------

        old_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(
            val_loss
        )

        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:

            print(
                f"\nLearning rate reduced:"
                f" {old_lr:.8f}"
                f" -> {new_lr:.8f}"
            )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            improvement = (
                best_val_loss - val_loss
                if best_val_loss != float("inf")
                else 0.0
            )

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                OUTPUT_CHECKPOINT
            )

            print(
                "\nNEW BEST MODEL SAVED!"
            )

            print(
                f"Validation loss: "
                f"{best_val_loss:.6f}"
            )

            if improvement > 0:

                print(
                    f"Improvement: "
                    f"{improvement:.6f}"
                )

            print(
                f"Checkpoint:"
                f"\n{OUTPUT_CHECKPOINT}"
            )

        else:

            epochs_without_improvement += 1

            print(
                "\nValidation loss did not improve."
            )

            print(
                f"Epochs without improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEARLY STOPPING TRIGGERED."
            )

            print(
                "Validation loss has not improved."
            )

            break

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("IMPROVED COMBINED TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"\nBest model saved at:"
        f"\n{OUTPUT_CHECKPOINT}"
    )

    print(
        "\nTraining datasets:"
    )

    for dataset_name in DATASETS:

        print(
            f"  - {dataset_name}"
        )

    print(
        "\nTest datasets remain completely separate."
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()