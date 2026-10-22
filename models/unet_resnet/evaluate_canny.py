import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parents[1]

PYTORCH_DIR = PROJECT_DIR / "pytorch"

sys.path.append(str(PYTORCH_DIR))
sys.path.append(str(CURRENT_DIR))

from canny_dataset import CannyDataset
from canny_model import create_canny_model


# ==================================================
# CONFIGURATION
# ==================================================

IMAGE_SIZE = 352
BATCH_SIZE = 4
NUM_WORKERS = 0

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

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "unet_resnet34_canny_extended_best.pth"
)


# ==================================================
# DICE + IOU
# ==================================================

def calculate_metrics(
    predictions,
    targets
):

    predictions = predictions.float()
    targets = targets.float()

    intersection = (
        predictions * targets
    ).sum(
        dim=(1, 2, 3)
    )

    prediction_area = predictions.sum(
        dim=(1, 2, 3)
    )

    target_area = targets.sum(
        dim=(1, 2, 3)
    )

    union = (
        prediction_area
        +
        target_area
        -
        intersection
    )

    dice = (
        (2.0 * intersection + 1e-7)
        /
        (prediction_area + target_area + 1e-7)
    )

    iou = (
        (intersection + 1e-7)
        /
        (union + 1e-7)
    )

    return dice.mean().item(), iou.mean().item()


# ==================================================
# LOAD MODEL
# ==================================================

print("=" * 60)
print("CANNY U-NET EVALUATION")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

print(
    "Loading checkpoint:",
    CHECKPOINT_PATH
)

model = create_canny_model()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")
print()


# ==================================================
# EVALUATE EACH DATASET
# ==================================================

results = {}


for dataset_name in DATASETS:

    print("=" * 60)
    print(f"Evaluating: {dataset_name}")
    print("=" * 60)

    dataset_dir = (
        PROJECT_DIR
        / "datasets"
        / dataset_name
    )

    test_dataset = CannyDataset(
        dataset_dir=dataset_dir,
        split="test",
        image_size=IMAGE_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    with torch.no_grad():

        for images, masks in test_loader:

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

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities > 0.5
            ).float()

            dice, iou = calculate_metrics(
                predictions,
                masks
            )

            total_dice += dice
            total_iou += iou

            total_batches += 1

    average_dice = (
        total_dice
        /
        total_batches
    )

    average_iou = (
        total_iou
        /
        total_batches
    )

    results[dataset_name] = (
        average_dice,
        average_iou
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    print(
        f"Test batches: {total_batches}"
    )

    print(
        f"Dice: {average_dice:.6f}"
    )

    print(
        f"IoU:  {average_iou:.6f}"
    )

    print()


# ==================================================
# FINAL SUMMARY
# ==================================================

print("=" * 60)
print("CANNY MODEL RESULTS")
print("=" * 60)

print(
    f"{'Dataset':<15}"
    f"{'Dice':>12}"
    f"{'IoU':>12}"
)

print("-" * 39)

for dataset_name, (dice, iou) in results.items():

    print(
        f"{dataset_name:<15}"
        f"{dice:>12.6f}"
        f"{iou:>12.6f}"
    )

print("=" * 60)