from pathlib import Path
from PIL import Image
from collections import Counter


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets" / "COD10K_CAM"


def analyze_split(split_name):

    print("\n" + "=" * 60)
    print(f"ANALYZING {split_name.upper()} SPLIT")
    print("=" * 60)

    image_dir = DATASET_DIR / split_name / "images"
    mask_dir = DATASET_DIR / split_name / "masks"

    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.is_file()
    )

    dimensions = []
    image_modes = Counter()
    mask_modes = Counter()
    mask_values = Counter()

    for index, image_path in enumerate(image_files):

        mask_path = (
            mask_dir
            / f"{image_path.stem}.png"
        )

        # Read image
        with Image.open(image_path) as image:

            dimensions.append(image.size)
            image_modes[image.mode] += 1

        # Read mask
        with Image.open(mask_path) as mask:

            mask_modes[mask.mode] += 1

            # Get unique pixel values
            values = mask.getdata()

            for value in set(values):
                mask_values[value] += 1

        # Progress indicator
        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(image_files)} images"
            )

    # ---------------------------------------------
    # DIMENSION STATISTICS
    # ---------------------------------------------

    widths = [
        width
        for width, height in dimensions
    ]

    heights = [
        height
        for width, height in dimensions
    ]

    print("\nIMAGE DIMENSIONS")

    print(
        f"Minimum width:  {min(widths)}"
    )

    print(
        f"Maximum width:  {max(widths)}"
    )

    print(
        f"Minimum height: {min(heights)}"
    )

    print(
        f"Maximum height: {max(heights)}"
    )

    print(
        f"Unique image sizes: "
        f"{len(set(dimensions))}"
    )

    print("\nIMAGE MODES")

    for mode, count in image_modes.items():
        print(f"{mode}: {count}")

    print("\nMASK MODES")

    for mode, count in mask_modes.items():
        print(f"{mode}: {count}")

    print("\nUNIQUE MASK PIXEL VALUES")

    for value in sorted(
        mask_values,
        key=lambda x: str(x)
    ):
        print(
            f"{value}: "
            f"present in "
            f"{mask_values[value]} masks"
        )


def main():

    print("=" * 60)
    print("COD10K DATASET ANALYSIS")
    print("=" * 60)

    analyze_split("train")
    analyze_split("val")
    analyze_split("test")


if __name__ == "__main__":
    main()