from pathlib import Path
import sys

import torch
import torch.nn as nn


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent

# train.py is inside:
# project/models/unet_resnet/
# So parent.parent is:
# project/
PROJECT_DIR = CURRENT_DIR.parent.parent

PYTORCH_DIR = PROJECT_DIR / "pytorch"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

sys.path.append(str(PYTORCH_DIR))


# ==================================================
# IMPORTS
# ==================================================

from dataloader import get_dataloaders
from model import create_model


# ==================================================
# CONFIGURATION
# ==================================================

# Number of ADDITIONAL epochs to train
EPOCHS = 20

# Lower learning rate for fine-tuning
LEARNING_RATE = 1e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# DICE LOSS
# ==================================================

def dice_loss(outputs, masks, smooth=1.0):

    # Convert raw logits to probabilities
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


# ==================================================
# COMBINED LOSS
# ==================================================

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


# ==================================================
# TRAIN ONE EPOCH
# ==================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    bce_criterion
):

    model.train()

    total_loss = 0.0

    for images, masks in loader:

        # Move data to GPU
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = combined_loss(
            outputs,
            masks,
            bce_criterion
        )

        # Backward pass
        loss.backward()

        # Update model
        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(loader)
    )

    return average_loss


# ==================================================
# VALIDATE
# ==================================================

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

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = combined_loss(
                outputs,
                masks,
                bce_criterion
            )

            total_loss += loss.item()

    average_loss = (
        total_loss / len(loader)
    )

    return average_loss


# ==================================================
# MAIN TRAINING
# ==================================================

def main():

    print("=" * 60)
    print("U-NET RESNET-34 CONTINUED TRAINING")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # ----------------------------------------------
    # Create checkpoint directory
    # ----------------------------------------------

    CHECKPOINT_DIR.mkdir(
        exist_ok=True
    )


    # ----------------------------------------------
    # Load data
    # ----------------------------------------------

    print("\nLoading dataloaders...")

    train_loader, val_loader, _ = (
        get_dataloaders()
    )

    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )


    # ----------------------------------------------
    # Create model
    # ----------------------------------------------

    print("\nCreating model...")

    model = create_model().to(
        DEVICE
    )


    # ----------------------------------------------
    # Load previous best checkpoint
    # ----------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / "unet_resnet34_best.pth"
    )

    print(
        "\nLoading previous best checkpoint..."
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    print(
        "Checkpoint loaded successfully."
    )


    # ----------------------------------------------
    # Loss
    # ----------------------------------------------

    bce_criterion = (
        nn.BCEWithLogitsLoss()
    )


    # ----------------------------------------------
    # Optimizer
    # ----------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ----------------------------------------------
    # Previous best validation loss
    # ----------------------------------------------

    best_val_loss = 0.621343

    print(
        f"\nPrevious best validation loss: "
        f"{best_val_loss:.6f}"
    )


    # ==============================================
    # TRAINING LOOP
    # ==============================================

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch "
            f"{epoch + 1}/{EPOCHS}"
        )

        print("-" * 60)


        # ------------------------------------------
        # Train
        # ------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            bce_criterion
        )


        # ------------------------------------------
        # Validate
        # ------------------------------------------

        val_loss = validate(
            model,
            val_loader,
            bce_criterion
        )


        print(
            f"Train Loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.6f}"
        )


        # ------------------------------------------
        # SAVE BEST MODEL
        # ------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = (
                CHECKPOINT_DIR
                / "unet_resnet34_best.pth"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                "NEW BEST MODEL SAVED!"
            )

            print(
                checkpoint_path
            )

        else:

            print(
                "Validation loss did not improve."
            )

            print(
                "Previous best model is kept."
            )


    # ==============================================
    # TRAINING COMPLETE
    # ==============================================

    print("\n" + "=" * 60)
    print("CONTINUED TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"\nBest Validation Loss: "
        f"{best_val_loss:.6f}"
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    main()