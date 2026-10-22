from pathlib import Path
import random
import shutil


# ==================================================
# CONFIGURATION
# ==================================================

RANDOM_SEED = 42
TRAIN_RATIO = 0.90


# ==================================================
# PATHS
# ==================================================

# This script is inside IVP_project/project/
PROJECT_DIR = Path(__file__).resolve().parent

# IVP_project/
ROOT_DIR = PROJECT_DIR.parent

# Original dataset - DO NOT MODIFY
ORIGINAL_DATASET = ROOT_DIR / "archive" / "COD10K-v3"

# New clean working dataset
OUTPUT_DATASET = PROJECT_DIR / "datasets" / "COD10K_CAM"


# ==================================================
# GET CAM IMAGE-MASK PAIRS
# ==================================================

def get_cam_pairs(split_dir):

    image_dir = split_dir / "Image"
    mask_dir = split_dir / "GT_Object"

    pairs = []

    # Select only CAM images
    cam_images = sorted(
        f for f in image_dir.iterdir()
        if f.is_file()
        and f.name.startswith("COD10K-CAM-")
    )

    for image_path in cam_images:

        # Match mask using filename stem
        mask_path = mask_dir / f"{image_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for image:\n{image_path}"
            )

        pairs.append((image_path, mask_path))

    return pairs


# ==================================================
# CREATE OUTPUT DIRECTORIES
# ==================================================

def create_output_directories():

    splits = ["train", "val", "test"]

    for split in splits:

        (OUTPUT_DATASET / split / "images").mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_DATASET / split / "masks").mkdir(
            parents=True,
            exist_ok=True
        )


# ==================================================
# COPY IMAGE-MASK PAIRS
# ==================================================

def copy_pairs(pairs, split_name):

    image_output_dir = OUTPUT_DATASET / split_name / "images"
    mask_output_dir = OUTPUT_DATASET / split_name / "masks"

    for image_path, mask_path in pairs:

        shutil.copy2(
            image_path,
            image_output_dir / image_path.name
        )

        shutil.copy2(
            mask_path,
            mask_output_dir / mask_path.name
        )


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("COD10K CAM DATASET PREPARATION")
    print("=" * 60)

    # Create directories
    create_output_directories()

    # ------------------------------------------------
    # TRAINING DATA
    # ------------------------------------------------

    print("\nLoading original training CAM pairs...")

    original_train_dir = ORIGINAL_DATASET / "Train"

    train_pairs = get_cam_pairs(original_train_dir)

    print(f"Total CAM training pairs found: {len(train_pairs)}")

    # Shuffle reproducibly
    random.seed(RANDOM_SEED)
    random.shuffle(train_pairs)

    # 90/10 split
    train_count = int(len(train_pairs) * TRAIN_RATIO)

    final_train_pairs = train_pairs[:train_count]
    val_pairs = train_pairs[train_count:]

    print(f"Training pairs:   {len(final_train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    # ------------------------------------------------
    # TEST DATA
    # ------------------------------------------------

    print("\nLoading official test CAM pairs...")

    original_test_dir = ORIGINAL_DATASET / "Test"

    test_pairs = get_cam_pairs(original_test_dir)

    print(f"Official test pairs: {len(test_pairs)}")

    # ------------------------------------------------
    # COPY FILES
    # ------------------------------------------------

    print("\nCopying training files...")
    copy_pairs(final_train_pairs, "train")

    print("Copying validation files...")
    copy_pairs(val_pairs, "val")

    print("Copying test files...")
    copy_pairs(test_pairs, "test")

    # ------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"\nOutput location:")
    print(OUTPUT_DATASET)

    print(f"\nTrain images: {len(final_train_pairs)}")
    print(f"Train masks:  {len(final_train_pairs)}")

    print(f"\nVal images:   {len(val_pairs)}")
    print(f"Val masks:    {len(val_pairs)}")

    print(f"\nTest images:  {len(test_pairs)}")
    print(f"Test masks:   {len(test_pairs)}")


if __name__ == "__main__":
    main()