from pathlib import Path

import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ==================================================
# PATH SETUP
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    CURRENT_DIR
    / "svm_data"
)

MODEL_DIR = (
    CURRENT_DIR
    / "checkpoints"
)

MODEL_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# CONFIGURATION
# ==================================================

RANDOM_SEED = 42


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("SVM TRAINING")
    print("=" * 60)


    # ----------------------------------------------
    # Load prepared data
    # ----------------------------------------------

    print("\nLoading training data...")

    X = np.load(
        DATA_DIR / "X_train.npy"
    )

    y = np.load(
        DATA_DIR / "y_train.npy"
    )

    print(
        f"Feature shape: {X.shape}"
    )

    print(
        f"Label shape: {y.shape}"
    )

    print(
        f"\nObject samples: "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Background samples: "
        f"{np.sum(y == 0)}"
    )


    # ----------------------------------------------
    # Create SVM pipeline
    # ----------------------------------------------

    print("\nCreating SVM model...")

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "svm",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=RANDOM_SEED
                )
            )
        ]
    )


    # ----------------------------------------------
    # Train
    # ----------------------------------------------

    print("\nTraining SVM...")

    model.fit(
        X,
        y
    )

    print(
        "Training completed successfully."
    )


    # ----------------------------------------------
    # Save model
    # ----------------------------------------------

    model_path = (
        MODEL_DIR
        / "svm_pixel_classifier.joblib"
    )

    print(
        "\nSaving model..."
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Model saved successfully."
    )


    # ==============================================
    # SUMMARY
    # ==============================================

    print("\n" + "=" * 60)
    print("SVM TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"\nModel saved at:"
    )

    print(
        model_path
    )


if __name__ == "__main__":
    main()