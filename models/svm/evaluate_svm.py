from pathlib import Path
import sys

import cv2
import joblib
import numpy as np

from PIL import Image


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = CURRENT_DIR.parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "datasets"
    / "COD10K_CAM"
)

TEST_IMAGE_DIR = (
    DATASET_DIR
    / "test"
    / "images"
)

TEST_MASK_DIR = (
    DATASET_DIR
    / "test"
    / "masks"
)

MODEL_PATH = (
    CURRENT_DIR
    / "checkpoints"
    / "svm_pixel_classifier.joblib"
)

RESULTS_DIR = (
    CURRENT_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# CONFIGURATION
# ==================================================

IMAGE_SIZE = 128


# ==================================================
# FEATURE EXTRACTION
# IMPORTANT:
# Must be identical to training.
# ==================================================

def extract_features(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

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
# METRICS
# ==================================================

def calculate_dice(
    prediction,
    ground_truth,
    smooth=1.0
):

    prediction = prediction.astype(
        np.float32
    )

    ground_truth = ground_truth.astype(
        np.float32
    )

    intersection = np.sum(
        prediction * ground_truth
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        np.sum(prediction)
        + np.sum(ground_truth)
        + smooth
    )

    return dice


def calculate_iou(
    prediction,
    ground_truth,
    smooth=1.0
):

    prediction = prediction.astype(
        bool
    )

    ground_truth = ground_truth.astype(
        bool
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("SVM TEST EVALUATION")
    print("=" * 60)


    # ----------------------------------------------
    # Load model
    # ----------------------------------------------

    print("\nLoading trained SVM model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )


    # ----------------------------------------------
    # Find test images
    # ----------------------------------------------

    image_paths = sorted(
        TEST_IMAGE_DIR.glob("*.jpg")
    )

    print(
        f"\nTest images found: "
        f"{len(image_paths)}"
    )


    dice_scores = []
    iou_scores = []


    # ==============================================
    # EVALUATE
    # ==============================================

    for index, image_path in enumerate(
        image_paths
    ):

        mask_path = (
            TEST_MASK_DIR
            / f"{image_path.stem}.png"
        )

        if not mask_path.exists():

            continue


        # ------------------------------------------
        # Load image
        # ------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = np.array(
            image
        )


        # ------------------------------------------
        # Load mask
        # ------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = np.array(
            mask
        )


        # ------------------------------------------
        # Resize image and mask
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

        original_shape = (
            features.shape[0],
            features.shape[1]
        )

        features = features.reshape(
            -1,
            features.shape[-1]
        )


        # ------------------------------------------
        # Predict pixels
        # ------------------------------------------

        predictions = model.predict(
            features
        )


        # ------------------------------------------
        # Reshape prediction into mask
        # ------------------------------------------

        predictions = predictions.reshape(
            original_shape
        )


        # ------------------------------------------
        # Ground truth
        # ------------------------------------------

        ground_truth = (
            mask > 127
        ).astype(
            np.uint8
        )


        # ------------------------------------------
        # Calculate metrics
        # ------------------------------------------

        dice = calculate_dice(
            predictions,
            ground_truth
        )

        iou = calculate_iou(
            predictions,
            ground_truth
        )

        dice_scores.append(
            dice
        )

        iou_scores.append(
            iou
        )


        # ------------------------------------------
        # Save first 10 predictions
        # ------------------------------------------

        if index < 10:

            prediction_image = (
                predictions * 255
            ).astype(
                np.uint8
            )

            output_path = (
                RESULTS_DIR
                / f"{image_path.stem}_prediction.png"
            )

            Image.fromarray(
                prediction_image
            ).save(
                output_path
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
    # RESULTS
    # ==============================================

    average_dice = np.mean(
        dice_scores
    )

    average_iou = np.mean(
        iou_scores
    )


    print("\n" + "=" * 60)
    print("SVM TEST RESULTS")
    print("=" * 60)

    print(
        f"\nAverage Dice Score: "
        f"{average_dice:.6f}"
    )

    print(
        f"Average IoU Score:  "
        f"{average_iou:.6f}"
    )

    print(
        f"\nPredicted masks saved to:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()