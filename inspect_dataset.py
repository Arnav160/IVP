from pathlib import Path


# --------------------------------------------------
# FIND PROJECT AND DATASET PATHS
# --------------------------------------------------

# This script is inside: IVP_project/project/
PROJECT_DIR = Path(__file__).resolve().parent

# Go one level up to IVP_project/
ROOT_DIR = PROJECT_DIR.parent

# Original COD10K dataset
DATASET_DIR = ROOT_DIR / "archive" / "COD10K-v3"

TRAIN_DIR = DATASET_DIR / "Train"
TEST_DIR = DATASET_DIR / "Test"


# --------------------------------------------------
# FUNCTION TO INSPECT A SPLIT
# --------------------------------------------------

def inspect_split(split_name, split_dir):

    image_dir = split_dir / "Image"
    mask_dir = split_dir / "GT_Object"

    # Get all files
    images = sorted([f for f in image_dir.iterdir() if f.is_file()])
    masks = sorted([f for f in mask_dir.iterdir() if f.is_file()])

    # Separate CAM and NonCAM images
    cam_images = [
    f for f in images
    if f.name.startswith("COD10K-CAM-")
    ]

    noncam_images = [
        f for f in images
        if f.name.startswith("COD10K-NonCAM-")
    ]

    print("\n" + "=" * 50)
    print(f"{split_name} DATASET")
    print("=" * 50)

    print(f"Image folder: {image_dir}")
    print(f"Mask folder:  {mask_dir}")

    print(f"\nTotal images:    {len(images)}")
    print(f"Total masks:     {len(masks)}")
    print(f"CAM images:      {len(cam_images)}")
    print(f"NonCAM images:   {len(noncam_images)}")

    # Create filename-stem sets
    image_stems = {f.stem for f in images}
    mask_stems = {f.stem for f in masks}

    missing_masks = image_stems - mask_stems
    missing_images = mask_stems - image_stems

    print(f"\nImages without matching mask: {len(missing_masks)}")
    print(f"Masks without matching image: {len(missing_images)}")

    # Show examples
    print("\nFirst 5 images:")
    for f in images[:5]:
        print(" ", f.name)

    print("\nFirst 5 masks:")
    for f in masks[:5]:
        print(" ", f.name)


# --------------------------------------------------
# RUN INSPECTION
# --------------------------------------------------

if __name__ == "__main__":

    print("COD10K DATASET INSPECTION")
    print(f"Dataset location: {DATASET_DIR}")

    inspect_split("TRAIN", TRAIN_DIR)
    inspect_split("TEST", TEST_DIR)