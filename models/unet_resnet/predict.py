from pathlib import Path
import sys
import random

import torch
import matplotlib.pyplot as plt


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = CURRENT_DIR.parent.parent

PYTORCH_DIR = PROJECT_DIR / "pytorch"

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "unet_resnet34_best.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "unet_resnet34"
)

sys.path.append(str(PYTORCH_DIR))


# ==================================================
# IMPORTS
# ==================================================

from dataloader import get_dataloaders
from model import create_model


# ==================================================
# DEVICE
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("U-NET RESNET-34 PREDICTION VISUALIZATION")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    # ----------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ----------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------

    print("\nCreating model...")

    model = create_model().to(
        DEVICE
    )

    print("Loading checkpoint...")

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print("Model ready.")

    # ----------------------------------------------
    # LOAD TEST DATA
    # ----------------------------------------------

    print("\nLoading test data...")

    _, _, test_loader = (
        get_dataloaders()
    )

    # ----------------------------------------------
    # SELECT RANDOM BATCH
    # ----------------------------------------------

    total_batches = len(test_loader)

    random_batch_index = random.randint(
        0,
        total_batches - 1
    )

    print(
        f"\nSelecting random batch: "
        f"{random_batch_index + 1}/"
        f"{total_batches}"
    )

    selected_images = None
    selected_masks = None

    for batch_index, (
        images,
        masks
    ) in enumerate(test_loader):

        if batch_index == random_batch_index:

            selected_images = images
            selected_masks = masks

            break

    # ----------------------------------------------
    # PREDICT
    # ----------------------------------------------

    images_gpu = selected_images.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            images_gpu
        )

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= 0.5
        ).float()

    # Move back to CPU
    predictions = predictions.cpu()

    # ----------------------------------------------
    # CREATE VISUALIZATION
    # ----------------------------------------------

    batch_size = selected_images.size(0)

    fig, axes = plt.subplots(
        batch_size,
        3,
        figsize=(12, batch_size * 4)
    )

    for i in range(batch_size):

        # ------------------------------------------
        # INPUT IMAGE
        # ------------------------------------------

        image = selected_images[
            i
        ].permute(
            1,
            2,
            0
        )

        image = image.cpu().numpy()

        # ------------------------------------------
        # UNNORMALIZE IMAGE
        # ------------------------------------------

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        )

        image = (
            image * std.numpy()
            + mean.numpy()
        )

        image = image.clip(
            0,
            1
        )

        # ------------------------------------------
        # GROUND TRUTH
        # ------------------------------------------

        ground_truth = selected_masks[
            i,
            0
        ].cpu().numpy()

        # ------------------------------------------
        # PREDICTION
        # ------------------------------------------

        prediction = predictions[
            i,
            0
        ].numpy()

        # ------------------------------------------
        # DISPLAY
        # ------------------------------------------

        axes[i, 0].imshow(
            image
        )

        axes[i, 0].set_title(
            "Input Image"
        )

        axes[i, 0].axis(
            "off"
        )

        axes[i, 1].imshow(
            ground_truth,
            cmap="gray"
        )

        axes[i, 1].set_title(
            "Ground Truth"
        )

        axes[i, 1].axis(
            "off"
        )

        axes[i, 2].imshow(
            prediction,
            cmap="gray"
        )

        axes[i, 2].set_title(
            "U-Net Prediction"
        )

        axes[i, 2].axis(
            "off"
        )

    # ----------------------------------------------
    # SAVE RESULT
    # ----------------------------------------------

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "prediction_samples.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    print("\nVisualization saved successfully.")

    print(
        f"\nSaved at:\n{output_path}"
    )

    plt.show()


if __name__ == "__main__":
    main()