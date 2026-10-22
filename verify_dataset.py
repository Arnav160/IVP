from pathlib import Path
from PIL import Image
import random


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_DIR
    / "datasets"
    / "COD10K_CAM"
)


# ==================================================
# VERIFY ONE SPLIT
# ==================================================

def verify_split(split_name):

    print("\n" + "=" * 60)
    print(f"VERIFYING {split_name.upper()} SPLIT")
    print("=" * 60)

    split_dir = DATASET_DIR / split_name

    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    images = sorted(
        f for f in image_dir.iterdir()
        if f.is_file()
    )

    masks = sorted(
        f for f in mask_dir.iterdir()
        if f.is_file()
    )

    print(f"\nImages found: {len(images)}")
    print(f"Masks found:  {len(masks)}")

    # ------------------------------------------------
    # CHECK ONLY CAM FILES
    # ------------------------------------------------

    non_cam_files = [
        f for f in images
        if not f.name.startswith("COD10K-CAM-")
    ]

    print(f"Non-CAM files found: {len(non_cam_files)}")

    # ------------------------------------------------
    # CHECK IMAGE-MASK PAIRING
    # ------------------------------------------------

    image_stems = {
        image.stem
        for image in images
    }

    mask_stems = {
        mask.stem
        for mask in masks
    }

    missing_masks = image_stems - mask_stems
    missing_images = mask_stems - image_stems

    print(
        f"Images without masks: {len(missing_masks)}"
    )

    print(
        f"Masks without images: {len(missing_images)}"
    )

    # ------------------------------------------------
    # OPEN RANDOM SAMPLES
    # ------------------------------------------------

    sample_count = min(5, len(images))

    samples = random.sample(
        images,
        sample_count
    )

    successful_loads = 0

    print("\nTesting random image-mask pairs:")

    for image_path in samples:

        mask_path = (
            mask_dir
            / f"{image_path.stem}.png"
        )

        try:

            with Image.open(image_path) as image:
                image.verify()

            with Image.open(mask_path) as mask:
                mask.verify()

            successful_loads += 1

            print(
                f"OK: {image_path.name}"
            )

        except Exception as error:

            print(
                f"ERROR: {image_path.name}"
            )

            print(error)

    print(
        f"\nSuccessfully loaded "
        f"{successful_loads}/{sample_count} "
        f"random pairs"
    )

    # ------------------------------------------------
    # RETURN FILE SETS FOR OVERLAP CHECK
    # ------------------------------------------------

    return image_stems


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("COD10K WORKING DATASET VERIFICATION")
    print("=" * 60)

    train_files = verify_split("train")

    val_files = verify_split("val")

    test_files = verify_split("test")

    # ------------------------------------------------
    # CHECK OVERLAP
    # ------------------------------------------------

    print("\n" + "=" * 60)
    print("CHECKING SPLIT OVERLAP")
    print("=" * 60)

    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files

    print(
        f"\nTrain-Val overlap: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train-Test overlap: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Val-Test overlap: "
        f"{len(val_test_overlap)}"
    )

    # ------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------

    print("\n" + "=" * 60)

    if (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    ):

        print(
            "DATASET VERIFICATION COMPLETED"
        )

    else:

        print(
            "WARNING: DATASET OVERLAP DETECTED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()