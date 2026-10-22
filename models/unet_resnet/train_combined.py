from pathlib import Path
import sys

import torch
import torch.nn as nn


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent.parent

PYTORCH_DIR = PROJECT_DIR / "pytorch"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

sys.path.append(str(PYTORCH_DIR))


# ==================================================
# IMPORTS
# ==================================================

from dataloader import get_combined_dataloaders
from model import create_model


# ==================================================
# CONFIGURATION
# ==================================================

DATASETS = [
    "COD10K_CAM",
    "CAMO",
    "NC4K"
]

# Number of epochs for this combined training run
EPOCHS = 20

# Start with a relatively small learning rate because
# we are starting from an already trained model.
LEARNING_RATE = 1e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# IMPORTANT:
# This is the known good COD10K-only checkpoint.
# We are NOT using the existing combined checkpoint.
STARTING_CHECKPOINT = (
    CHECKPOINT_DIR
    / "unet_resnet34_best.pth"
)

# New checkpoint name.
# This will NOT overwrite either existing checkpoint.
BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "unet_resnet34_all_datasets_best.pth"
)


# ==================================================
# DICE LOSS
# ==================================================

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

    return total_loss / len(loader)


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

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks,
                bce_criterion
            )

            total_loss += loss.item()

    return total_loss / len(loader)


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print("U-NET RESNET-34 COMBINED DATASET TRAINING")
    print("=" * 70)

    # ------------------------------------------------
    # DEVICE
    # ------------------------------------------------

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

        print(
            "Training will run on CPU."
        )


    # ------------------------------------------------
    # CHECK STARTING CHECKPOINT
    # ------------------------------------------------

    print(
        "\nChecking starting checkpoint..."
    )

    print(
        f"Starting checkpoint:\n"
        f"{STARTING_CHECKPOINT}"
    )

    if not STARTING_CHECKPOINT.exists():

        raise FileNotFoundError(
            "\nStarting checkpoint not found:\n"
            f"{STARTING_CHECKPOINT}"
        )

    print(
        "Starting checkpoint found."
    )


    # ------------------------------------------------
    # CHECK OUTPUT CHECKPOINT
    # ------------------------------------------------

    print(
        "\nOutput checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    if BEST_CHECKPOINT.exists():

        print(
            "\nWARNING:"
        )

        print(
            "The output checkpoint already exists."
        )

        print(
            "This script will overwrite it if training"
            " produces a new best model."
        )


    # ------------------------------------------------
    # LOAD COMBINED DATA
    # ------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "LOADING COMBINED DATASETS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDatasets:"
    )

    for dataset in DATASETS:

        print(
            f"  - {dataset}"
        )


    print(
        "\nCreating combined dataloaders..."
    )

    (
        train_loader,
        val_loader,
        test_loaders
    ) = get_combined_dataloaders(
        DATASETS
    )


    print(
        "\nCombined training samples:"
        f" {len(train_loader.dataset)}"
    )

    print(
        "Combined validation samples:"
        f" {len(val_loader.dataset)}"
    )

    print(
        "\nExpected training samples: 6832"
    )

    print(
        "Expected validation samples: 916"
    )


    # ------------------------------------------------
    # VERIFY DATA COUNTS
    # ------------------------------------------------

    if len(train_loader.dataset) != 6832:

        raise RuntimeError(
            "Unexpected combined training dataset size."
        )

    if len(val_loader.dataset) != 916:

        raise RuntimeError(
            "Unexpected combined validation dataset size."
        )

    print(
        "\nDataset counts verified."
    )


    # ------------------------------------------------
    # MODEL
    # ------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING MODEL"
    )

    print(
        "=" * 70
    )

    model = create_model().to(
        DEVICE
    )

    print(
        "Model created successfully."
    )


    # ------------------------------------------------
    # LOAD GOOD ORIGINAL CHECKPOINT
    # ------------------------------------------------

    print(
        "\nLoading original COD10K best checkpoint..."
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
        "\nTraining will START FROM:"
    )

    print(
        "unet_resnet34_best.pth"
    )

    print(
        "\nThe existing combined checkpoint"
    )

    print(
        "unet_resnet34_combined_best.pth"
    )

    print(
        "will NOT be used."
    )


    # ------------------------------------------------
    # LOSS
    # ------------------------------------------------

    bce_criterion = (
        nn.BCEWithLogitsLoss()
    )


    # ------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ------------------------------------------------
    # BEST VALIDATION LOSS
    # ------------------------------------------------

    best_val_loss = float("inf")


    # ------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING COMBINED TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        f"\nEpochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )


    for epoch in range(EPOCHS):

        print(
            "\n" + "-" * 70
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            "-" * 70
        )


        # --------------------------------------------
        # TRAIN
        # --------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            bce_criterion
        )


        # --------------------------------------------
        # VALIDATE
        # --------------------------------------------

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


        # --------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                BEST_CHECKPOINT
            )

            print(
                "\nNEW BEST MODEL SAVED!"
            )

            print(
                BEST_CHECKPOINT
            )

        else:

            print(
                "\nValidation loss did not improve."
            )

            print(
                "Previous best model is kept."
            )


    # ------------------------------------------------
    # COMPLETE
    # ------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "COMBINED TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest validation loss:"
        f" {best_val_loss:.6f}"
    )

    print(
        "\nBest model saved at:"
    )

    print(
        BEST_CHECKPOINT
    )

    print(
        "\nTraining datasets:"
    )

    for dataset in DATASETS:

        print(
            f"  - {dataset}"
        )

    print(
        "\nThe three test datasets remain"
        " completely separate."
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()