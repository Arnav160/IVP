import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, ConcatDataset


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parents[1]

PYTORCH_DIR = PROJECT_DIR / "pytorch"

sys.path.append(str(PYTORCH_DIR))
sys.path.append(str(CURRENT_DIR))


# ==================================================
# IMPORTS
# ==================================================

from canny_dataset import CannyDataset
from canny_model import create_canny_model


# ==================================================
# CONFIGURATION
# ==================================================

IMAGE_SIZE = 352
BATCH_SIZE = 4

NUM_WORKERS = 0

LEARNING_RATE = 1e-5

# Continue for 10 additional epochs
ADDITIONAL_EPOCHS = 10

DATASETS = [
    "COD10K_CAM",
    "CAMO",
    "NC4K"
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# CHECKPOINTS
# ==================================================

CHECKPOINT_DIR = (
    PROJECT_DIR
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Existing best model
START_CHECKPOINT = (
    CHECKPOINT_DIR
    / "unet_resnet34_canny_best.pth"
)

# New checkpoint for the extended experiment
EXTENDED_CHECKPOINT = (
    CHECKPOINT_DIR
    / "unet_resnet34_canny_extended_best.pth"
)


# ==================================================
# DEVICE
# ==================================================

print("=" * 60)
print("CANNY U-NET EXTENDED TRAINING")
print("=" * 60)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ==================================================
# LOAD DATASETS
# ==================================================

print("Loading datasets...")

train_datasets = []
val_datasets = []

for dataset_name in DATASETS:

    dataset_dir = (
        PROJECT_DIR
        / "datasets"
        / dataset_name
    )

    print(
        f"Loading {dataset_name}..."
    )

    train_dataset = CannyDataset(
        dataset_dir=dataset_dir,
        split="train",
        image_size=IMAGE_SIZE
    )

    val_dataset = CannyDataset(
        dataset_dir=dataset_dir,
        split="val",
        image_size=IMAGE_SIZE
    )

    train_datasets.append(
        train_dataset
    )

    val_datasets.append(
        val_dataset
    )

    print(
        f"  Train: {len(train_dataset)}"
    )

    print(
        f"  Val:   {len(val_dataset)}"
    )


# ==================================================
# COMBINE DATASETS
# ==================================================

combined_train_dataset = ConcatDataset(
    train_datasets
)

combined_val_dataset = ConcatDataset(
    val_datasets
)

print()
print(
    "Combined train:",
    len(combined_train_dataset)
)

print(
    "Combined val:",
    len(combined_val_dataset)
)


# ==================================================
# DATALOADERS
# ==================================================

train_loader = DataLoader(
    combined_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    combined_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(
    "Train batches:",
    len(train_loader)
)

print(
    "Val batches:",
    len(val_loader)
)

print()


# ==================================================
# MODEL
# ==================================================

print("Creating Canny U-Net...")

model = create_canny_model()

# --------------------------------------------------
# LOAD EXISTING BEST CHECKPOINT
# --------------------------------------------------

if not START_CHECKPOINT.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n{START_CHECKPOINT}"
    )

print(
    "Loading existing best checkpoint:"
)

print(
    START_CHECKPOINT
)

checkpoint = torch.load(
    START_CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint
)

model = model.to(
    DEVICE
)

print(
    "Existing Canny model loaded."
)

print()


# ==================================================
# DICE LOSS
# ==================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.contiguous()
        targets = targets.contiguous()

        intersection = (
            probabilities * targets
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probabilities.sum(
                dim=(1, 2, 3)
            )
            +
            targets.sum(
                dim=(1, 2, 3)
            )
        )

        dice = (
            (2.0 * intersection + self.smooth)
            /
            (denominator + self.smooth)
        )

        return (
            1.0 - dice.mean()
        )


# ==================================================
# LOSS
# ==================================================

bce_loss = nn.BCEWithLogitsLoss()

dice_loss = DiceLoss()


# ==================================================
# OPTIMIZER
# ==================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# EXTENDED TRAINING
# ==================================================

best_val_loss = float("inf")

for epoch in range(
    1,
    ADDITIONAL_EPOCHS + 1
):

    # ----------------------------------------------
    # TRAIN
    # ----------------------------------------------

    model.train()

    total_train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        bce = bce_loss(
            outputs,
            masks
        )

        dice = dice_loss(
            outputs,
            masks
        )

        loss = bce + dice

        loss.backward()

        optimizer.step()

        total_train_loss += (
            loss.item()
        )

    average_train_loss = (
        total_train_loss
        /
        len(train_loader)
    )


    # ----------------------------------------------
    # VALIDATION
    # ----------------------------------------------

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            )

            bce = bce_loss(
                outputs,
                masks
            )

            dice = dice_loss(
                outputs,
                masks
            )

            loss = bce + dice

            total_val_loss += (
                loss.item()
            )

    average_val_loss = (
        total_val_loss
        /
        len(val_loader)
    )


    # ----------------------------------------------
    # PRINT
    # ----------------------------------------------

    print(
        f"Extended Epoch {epoch:02d}/10 | "
        f"Train Loss: {average_train_loss:.6f} | "
        f"Val Loss: {average_val_loss:.6f}"
    )


    # ----------------------------------------------
    # SAVE BEST EXTENDED MODEL
    # ----------------------------------------------

    if average_val_loss < best_val_loss:

        best_val_loss = (
            average_val_loss
        )

        torch.save(
            model.state_dict(),
            EXTENDED_CHECKPOINT
        )

        print(
            f"  NEW EXTENDED BEST MODEL SAVED "
            f"(Val Loss: {best_val_loss:.6f})"
        )


# ==================================================
# COMPLETE
# ==================================================

print()
print("=" * 60)
print("EXTENDED TRAINING COMPLETE")
print("=" * 60)

print(
    "Best extended validation loss:",
    f"{best_val_loss:.6f}"
)

print(
    "Extended checkpoint:",
    EXTENDED_CHECKPOINT
)