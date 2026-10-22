import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

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

from cod10k_dataset import COD10KDataset
from canny_dataset import CannyDataset

from model import create_model
from canny_model import create_canny_model


# ==================================================
# CONFIG
# ==================================================

IMAGE_SIZE = 352

NUM_IMAGES = 10

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

RGB_CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "unet_resnet34_all_datasets_best.pth"
)

CANNY_CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "unet_resnet34_canny_best.pth"
)


# ==================================================
# OUTPUT DIRECTORY
# ==================================================

OUTPUT_DIR = (
    CURRENT_DIR
    / "rgb_vs_canny_visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# NORMALIZATION
# ==================================================

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


# ==================================================
# DENORMALIZE RGB
# ==================================================

def denormalize_rgb(image):

    image = image.cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    image = (
        image * STD
        + MEAN
    )

    image = np.clip(
        image,
        0,
        1
    )

    return image


# ==================================================
# LOAD RGB MODEL
# ==================================================

print("=" * 60)
print("LOADING RGB MODEL")
print("=" * 60)

rgb_model = create_model()

rgb_checkpoint = torch.load(
    RGB_CHECKPOINT,
    map_location=DEVICE
)

rgb_model.load_state_dict(
    rgb_checkpoint
)

rgb_model = rgb_model.to(DEVICE)

rgb_model.eval()

print("RGB model loaded.")


# ==================================================
# LOAD CANNY MODEL
# ==================================================

print()
print("=" * 60)
print("LOADING CANNY MODEL")
print("=" * 60)

canny_model = create_canny_model()

canny_checkpoint = torch.load(
    CANNY_CHECKPOINT,
    map_location=DEVICE
)

canny_model.load_state_dict(
    canny_checkpoint
)

canny_model = canny_model.to(DEVICE)

canny_model.eval()

print("Canny model loaded.")
print()


# ==================================================
# PROCESS DATASETS
# ==================================================

for dataset_name in DATASETS:

    print(
        f"Processing {dataset_name}..."
    )

    dataset_dir = (
        PROJECT_DIR
        / "datasets"
        / dataset_name
    )

    # ----------------------------------------------
    # RGB DATASET
    # ----------------------------------------------

    rgb_dataset = COD10KDataset(
        dataset_dir=dataset_dir,
        split="test",
        image_size=IMAGE_SIZE
    )

    # ----------------------------------------------
    # CANNY DATASET
    # ----------------------------------------------

    canny_dataset = CannyDataset(
        dataset_dir=dataset_dir,
        split="test",
        image_size=IMAGE_SIZE
    )

    # ----------------------------------------------
    # OUTPUT DIRECTORY
    # ----------------------------------------------

    dataset_output_dir = (
        OUTPUT_DIR
        / dataset_name
    )

    dataset_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Select evenly spaced images

    indices = np.linspace(
        0,
        len(rgb_dataset) - 1,
        NUM_IMAGES,
        dtype=int
    )

    for image_number, index in enumerate(
        indices,
        start=1
    ):

        # ==========================================
        # GET SAME IMAGE
        # ==========================================

        rgb_image_tensor, rgb_mask = (
            rgb_dataset[index]
        )

        canny_image_tensor, canny_mask = (
            canny_dataset[index]
        )

        # ------------------------------------------
        # RGB INPUT
        # ------------------------------------------

        rgb_input = (
            rgb_image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ------------------------------------------
        # CANNY INPUT
        # ------------------------------------------

        canny_input = (
            canny_image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ==========================================
        # RGB MODEL PREDICTION
        # ==========================================

        with torch.no_grad():

            rgb_output = rgb_model(
                rgb_input
            )

            rgb_probability = torch.sigmoid(
                rgb_output
            )

            rgb_prediction = (
                rgb_probability > 0.5
            ).float()

        # ==========================================
        # CANNY MODEL PREDICTION
        # ==========================================

        with torch.no_grad():

            canny_output = canny_model(
                canny_input
            )

            canny_probability = torch.sigmoid(
                canny_output
            )

            canny_prediction = (
                canny_probability > 0.5
            ).float()

        # ==========================================
        # CONVERT TO NUMPY
        # ==========================================

        original_image = denormalize_rgb(
            rgb_image_tensor
        )

        ground_truth = (
            rgb_mask[0]
            .cpu()
            .numpy()
        )

        rgb_pred = (
            rgb_prediction[0, 0]
            .cpu()
            .numpy()
        )

        canny_pred = (
            canny_prediction[0, 0]
            .cpu()
            .numpy()
        )

        # ==========================================
        # CREATE COMPARISON
        # ==========================================

        fig, axes = plt.subplots(
            2,
            4,
            figsize=(16, 8)
        )

        # ------------------------------------------
        # ROW 1
        # ------------------------------------------

        axes[0, 0].imshow(
            original_image
        )

        axes[0, 0].set_title(
            "Original Image"
        )

        axes[0, 1].imshow(
            ground_truth,
            cmap="gray"
        )

        axes[0, 1].set_title(
            "Ground Truth"
        )

        axes[0, 2].imshow(
            rgb_pred,
            cmap="gray"
        )

        axes[0, 2].set_title(
            "RGB / ResNet Prediction"
        )

        axes[0, 3].imshow(
            canny_pred,
            cmap="gray"
        )

        axes[0, 3].set_title(
            "RGB + Canny Prediction"
        )

        # ------------------------------------------
        # ROW 2 - OVERLAYS
        # ------------------------------------------

        axes[1, 0].imshow(
            original_image
        )

        axes[1, 0].set_title(
            "Original"
        )

        axes[1, 1].imshow(
            original_image
        )

        axes[1, 1].imshow(
            ground_truth,
            cmap="gray",
            alpha=0.4
        )

        axes[1, 1].set_title(
            "Ground Truth Overlay"
        )

        axes[1, 2].imshow(
            original_image
        )

        axes[1, 2].imshow(
            rgb_pred,
            cmap="gray",
            alpha=0.4
        )

        axes[1, 2].set_title(
            "RGB / ResNet Overlay"
        )

        axes[1, 3].imshow(
            original_image
        )

        axes[1, 3].imshow(
            canny_pred,
            cmap="gray",
            alpha=0.4
        )

        axes[1, 3].set_title(
            "RGB + Canny Overlay"
        )

        # ------------------------------------------
        # CLEAN AXES
        # ------------------------------------------

        for ax in axes.flat:

            ax.axis("off")

        fig.suptitle(
            f"{dataset_name} - Sample {image_number}",
            fontsize=16
        )

        plt.tight_layout()

        # ------------------------------------------
        # SAVE
        # ------------------------------------------

        output_path = (
            dataset_output_dir
            / f"comparison_{image_number:02d}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

    print(
        f"  Saved {NUM_IMAGES} comparisons."
    )


# ==================================================
# COMPLETE
# ==================================================

print()
print("=" * 60)
print("RGB vs CANNY COMPARISON COMPLETE")
print("=" * 60)

print(
    "Output folder:",
    OUTPUT_DIR
)