import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

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
# CONFIG
# ==================================================

IMAGE_SIZE = 352

NUM_IMAGES = 10

DATASETS = [
    "COD10K_CAM",
    "CAMO",
    "NC4K"
]

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "unet_resnet34_canny_best.pth"
)

OUTPUT_DIR = (
    CURRENT_DIR
    / "canny_prediction_visualizations"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# OUTPUT DIRECTORY
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# MODEL
# ==================================================

print("=" * 60)
print("CANNY PREDICTION VISUALIZATION")
print("=" * 60)

print("Device:", DEVICE)

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

print("Model loaded.")
print()


# ==================================================
# IMAGE NORMALIZATION VALUES
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
# GENERATE VISUALIZATIONS
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

    test_dataset = CannyDataset(
        dataset_dir=dataset_dir,
        split="test",
        image_size=IMAGE_SIZE
    )

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
        len(test_dataset) - 1,
        NUM_IMAGES,
        dtype=int
    )

    for image_number, index in enumerate(
        indices,
        start=1
    ):

        image_tensor, mask_tensor = (
            test_dataset[index]
        )

        # Add batch dimension
        input_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ------------------------------------------
        # MODEL PREDICTION
        # ------------------------------------------

        with torch.no_grad():

            output = model(
                input_tensor
            )

            probability = torch.sigmoid(
                output
            )

            prediction = (
                probability > 0.5
            ).float()

        # ------------------------------------------
        # EXTRACT RGB + CANNY
        # ------------------------------------------

        rgb_tensor = (
            image_tensor[:3]
        )

        canny_tensor = (
            image_tensor[3]
        )

        rgb_image = denormalize_rgb(
            rgb_tensor
        )

        canny_image = (
            canny_tensor
            .cpu()
            .numpy()
        )

        ground_truth = (
            mask_tensor[0]
            .cpu()
            .numpy()
        )

        predicted_mask = (
            prediction[0, 0]
            .cpu()
            .numpy()
        )

        # ------------------------------------------
        # CREATE FIGURE
        # ------------------------------------------

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4)
        )

        axes[0].imshow(
            rgb_image
        )

        axes[0].set_title(
            "Original RGB"
        )

        axes[1].imshow(
            canny_image,
            cmap="gray"
        )

        axes[1].set_title(
            "Canny Edge"
        )

        axes[2].imshow(
            ground_truth,
            cmap="gray"
        )

        axes[2].set_title(
            "Ground Truth"
        )

        axes[3].imshow(
            predicted_mask,
            cmap="gray"
        )

        axes[3].set_title(
            "Canny Prediction"
        )

        axes[4].imshow(
            rgb_image
        )

        axes[4].imshow(
            predicted_mask,
            cmap="jet",
            alpha=0.35
        )

        axes[4].set_title(
            "Prediction Overlay"
        )

        for ax in axes:

            ax.axis("off")

        fig.suptitle(
            f"{dataset_name} - Sample {image_number}",
            fontsize=14
        )

        plt.tight_layout()

        output_path = (
            dataset_output_dir
            / f"sample_{image_number:02d}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

    print(
        f"  Saved {NUM_IMAGES} visualizations."
    )


print()
print("=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print(
    "Output:",
    OUTPUT_DIR
)