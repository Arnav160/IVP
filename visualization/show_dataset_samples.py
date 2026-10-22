from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# ==========================================
# PATHS
# ==========================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = CURRENT_DIR.parent

DATASET_DIR = (
    PROJECT_DIR
    / "datasets"
    / "COD10K_CAM"
)


# ==========================================
# CONFIGURATION
# ==========================================

SPLIT = "train"
NUM_SAMPLES = 5


# ==========================================
# LOAD DATA
# ==========================================

def main():

    image_dir = (
        DATASET_DIR
        / SPLIT
        / "images"
    )

    mask_dir = (
        DATASET_DIR
        / SPLIT
        / "masks"
    )

    image_paths = sorted(
        [
            f
            for f in image_dir.iterdir()
            if f.is_file()
        ]
    )

    samples = random.sample(
        image_paths,
        NUM_SAMPLES
    )

    # ==========================================
    # CREATE FIGURE
    # ==========================================

    fig, axes = plt.subplots(
        NUM_SAMPLES,
        2,
        figsize=(10, NUM_SAMPLES * 5)
    )

    for row, image_path in enumerate(samples):

        mask_path = (
            mask_dir
            / f"{image_path.stem}.png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        # Original image
        axes[row, 0].imshow(image)

        axes[row, 0].set_title(
            "Camouflaged Object Image"
        )

        axes[row, 0].axis("off")

        # Ground truth mask
        axes[row, 1].imshow(
            mask,
            cmap="gray"
        )

        axes[row, 1].set_title(
            "Ground Truth Mask"
        )

        axes[row, 1].axis("off")

    plt.tight_layout()

    # ==========================================
    # SAVE FIGURE
    # ==========================================

    output_path = (
        CURRENT_DIR
        / "dataset_samples.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    print("\nDataset visualization created.")

    print(
        f"Saved at:\n{output_path}"
    )

    plt.show()


if __name__ == "__main__":
    main()