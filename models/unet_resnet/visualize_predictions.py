from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


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

from dataloader import get_dataloaders
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR /
    "unet_resnet34_all_datasets_best.pth"
)

DATASETS = [
    "COD10K_CAM",
    "CAMO",
    "NC4K"
]

# Number of images to visualize from each dataset
NUM_IMAGES = 10

# Prediction threshold
THRESHOLD = 0.5

OUTPUT_DIR = (
    CURRENT_DIR /
    "prediction_visualizations"
)


# ============================================================
# IMAGE DENORMALIZATION
# ============================================================

# ImageNet normalization used by the dataloader
MEAN = np.array([
    0.485,
    0.456,
    0.406
])

STD = np.array([
    0.229,
    0.224,
    0.225
])


def denormalize(image):

    image = image.cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    image = (
        image * STD
    ) + MEAN

    image = np.clip(
        image,
        0,
        1
    )

    return image


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def save_visualization(
    image,
    ground_truth,
    prediction,
    output_path,
    dataset_name,
    image_number
):

    # Convert image back to normal RGB
    image = denormalize(image)

    # Convert masks to numpy
    ground_truth = (
        ground_truth
        .cpu()
        .numpy()
        .squeeze()
    )

    prediction = (
        prediction
        .cpu()
        .numpy()
        .squeeze()
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(16, 4)
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    ax1 = figure.add_subplot(
        1, 4, 1
    )

    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ax2 = figure.add_subplot(
        1, 4, 2
    )

    ax2.imshow(
        ground_truth,
        cmap="gray"
    )

    ax2.set_title("Ground Truth")
    ax2.axis("off")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    ax3 = figure.add_subplot(
        1, 4, 3
    )

    ax3.imshow(
        prediction,
        cmap="gray"
    )

    ax3.set_title("U-Net Prediction")
    ax3.axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    ax4 = figure.add_subplot(
        1, 4, 4
    )

    ax4.imshow(image)

    ax4.imshow(
        prediction,
        cmap="jet",
        alpha=0.45
    )

    ax4.set_title("Prediction Overlay")
    ax4.axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    figure.suptitle(
        f"{dataset_name} - Test Image {image_number}",
        fontsize=14
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# VISUALIZE ONE DATASET
# ============================================================

def visualize_dataset(
    model,
    dataset_name
):

    print("\n" + "=" * 70)
    print(
        f"VISUALIZING {dataset_name} TEST DATASET"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    dataset_output_dir = (
        OUTPUT_DIR /
        dataset_name
    )

    dataset_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load test dataloader
    # --------------------------------------------------------

    print(
        f"\nLoading {dataset_name} test dataloader..."
    )

    _, _, test_loader = get_dataloaders(
        dataset_name=dataset_name
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    image_count = 0

    print(
        f"\nGenerating {NUM_IMAGES} visualizations..."
    )

    # --------------------------------------------------------
    # Run predictions
    # --------------------------------------------------------

    with torch.no_grad():

        for images, masks in test_loader:

            # Move images to GPU
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            # Model prediction
            outputs = model(
                images
            )

            # Convert logits to probabilities
            probabilities = torch.sigmoid(
                outputs
            )

            # Convert probabilities to binary masks
            predictions = (
                probabilities >= THRESHOLD
            ).float()

            # ------------------------------------------------
            # Save individual examples
            # ------------------------------------------------

            for batch_index in range(
                images.size(0)
            ):

                if image_count >= NUM_IMAGES:
                    break

                image_count += 1

                output_path = (
                    dataset_output_dir /
                    f"prediction_{image_count:03d}.png"
                )

                save_visualization(
                    images[batch_index],
                    masks[batch_index],
                    predictions[batch_index],
                    output_path,
                    dataset_name,
                    image_count
                )

                print(
                    f"Saved "
                    f"{image_count}/{NUM_IMAGES}: "
                    f"{output_path.name}"
                )

            if image_count >= NUM_IMAGES:
                break

    print(
        f"\nCompleted {dataset_name}."
    )

    print(
        f"Visualizations saved to:"
    )

    print(
        dataset_output_dir
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "U-NET RESNET-34 PREDICTION VISUALIZATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "CUDA version: "
            f"{torch.version.cuda}"
        )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    print(
        "\nChecking checkpoint..."
    )

    print(
        f"Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    print(
        "Checkpoint found."
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print(
        "\nCreating U-Net ResNet-34 model..."
    )

    model = create_model().to(
        DEVICE
    )

    print(
        "Model created successfully."
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading trained checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # Visualize all datasets
    # --------------------------------------------------------

    for dataset_name in DATASETS:

        visualize_dataset(
            model,
            dataset_name
        )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "VISUALIZATION COMPLETED"
    )
    print("=" * 70)

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nDatasets visualized:"
    )

    for dataset_name in DATASETS:

        print(
            f"  - {dataset_name}"
        )

    print(
        f"\nImages per dataset: "
        f"{NUM_IMAGES}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()