from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent

# svm -> models -> project
PROJECT_DIR = CURRENT_DIR.parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "datasets"
    / "COD10K_CAM"
)

TRAIN_IMAGE_DIR = (
    DATASET_DIR
    / "train"
    / "images"
)

TRAIN_MASK_DIR = (
    DATASET_DIR
    / "train"
    / "masks"
)

OUTPUT_DIR = (
    CURRENT_DIR
    / "svm_data"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# CONFIGURATION
# ==================================================

# Small size for classical ML processing
IMAGE_SIZE = 128

# Number of object/background pixels sampled
# from each image
SAMPLES_PER_CLASS = 100

# Limit for initial experiment.
# Set to None later if needed.
MAX_IMAGES = 1000

RANDOM_SEED = 42


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_features(image):

    # RGB image already provided

    # Convert RGB -> HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    # Convert RGB -> grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Sobel gradients
    grad_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = np.sqrt(
        grad_x ** 2
        + grad_y ** 2
    )

    # Stack features for every pixel
    features = np.concatenate(
        [
            image,
            hsv,
            gradient_magnitude[..., None]
        ],
        axis=2
    )

    return features


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("PREPARING SVM DATASET")
    print("=" * 60)

    np.random.seed(
        RANDOM_SEED
    )

    # ----------------------------------------------
    # Find images
    # ----------------------------------------------

    image_paths = sorted(
        TRAIN_IMAGE_DIR.glob("*.jpg")
    )

    print(
        f"\nTraining images found: "
        f"{len(image_paths)}"
    )

    if MAX_IMAGES is not None:

        image_paths = image_paths[
            :MAX_IMAGES
        ]

        print(
            f"Images selected: "
            f"{len(image_paths)}"
        )

    all_features = []
    all_labels = []


    # ==============================================
    # PROCESS IMAGES
    # ==============================================

    for index, image_path in enumerate(
        image_paths
    ):

        # Corresponding mask
        mask_path = (
            TRAIN_MASK_DIR
            / f"{image_path.stem}.png"
        )

        if not mask_path.exists():

            print(
                f"WARNING: Mask not found: "
                f"{mask_path.name}"
            )

            continue


        # ------------------------------------------
        # Load image
        # ------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = np.array(image)


        # ------------------------------------------
        # Load mask
        # ------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = np.array(mask)


        # ------------------------------------------
        # Resize
        # ------------------------------------------

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )


        # ------------------------------------------
        # Extract features
        # ------------------------------------------

        features = extract_features(
            image
        )

        features = features.reshape(
            -1,
            features.shape[-1]
        )


        # ------------------------------------------
        # Flatten mask
        # ------------------------------------------

        labels = (
            mask.reshape(-1) > 127
        ).astype(np.uint8)


        # ------------------------------------------
        # Find object/background pixels
        # ------------------------------------------

        object_indices = np.where(
            labels == 1
        )[0]

        background_indices = np.where(
            labels == 0
        )[0]


        # Skip unusual images
        if (
            len(object_indices) == 0
            or len(background_indices) == 0
        ):

            continue


        # ------------------------------------------
        # Sample object pixels
        # ------------------------------------------

        object_sample_count = min(
            SAMPLES_PER_CLASS,
            len(object_indices)
        )

        sampled_object = np.random.choice(
            object_indices,
            size=object_sample_count,
            replace=False
        )


        # ------------------------------------------
        # Sample background pixels
        # ------------------------------------------

        background_sample_count = min(
            SAMPLES_PER_CLASS,
            len(background_indices)
        )

        sampled_background = np.random.choice(
            background_indices,
            size=background_sample_count,
            replace=False
        )


        # ------------------------------------------
        # Combine selected pixels
        # ------------------------------------------

        selected_indices = np.concatenate(
            [
                sampled_object,
                sampled_background
            ]
        )

        selected_features = (
            features[selected_indices]
        )

        selected_labels = (
            labels[selected_indices]
        )


        all_features.append(
            selected_features
        )

        all_labels.append(
            selected_labels
        )


        # ------------------------------------------
        # Progress
        # ------------------------------------------

        if (
            (index + 1) % 100 == 0
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(image_paths)} images"
            )


    # ==============================================
    # COMBINE DATA
    # ==============================================

    print(
        "\nCombining SVM training samples..."
    )

    X = np.vstack(
        all_features
    ).astype(np.float32)

    y = np.concatenate(
        all_labels
    ).astype(np.uint8)


    # ==============================================
    # SAVE DATA
    # ==============================================

    print(
        "\nSaving data..."
    )

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y
    )


    # ==============================================
    # SUMMARY
    # ==============================================

    print("\n" + "=" * 60)
    print("SVM DATA PREPARATION COMPLETED")
    print("=" * 60)

    print(
        f"\nFeature shape: "
        f"{X.shape}"
    )

    print(
        f"Label shape: "
        f"{y.shape}"
    )

    print(
        f"\nObject samples: "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Background samples: "
        f"{np.sum(y == 0)}"
    )

    print(
        f"\nSaved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()