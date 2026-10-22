from pathlib import Path
import sys

import torch


# Allow imports from the current pytorch folder
CURRENT_DIR = Path(__file__).resolve().parent

sys.path.append(
    str(CURRENT_DIR)
)


from cod10k_dataset import COD10KDataset


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = (
    Path(__file__).resolve()
    .parent
    .parent
)

DATASET_DIR = (
    PROJECT_DIR
    / "datasets"
    / "COD10K_CAM"
)


# ==================================================
# TEST DATASETS
# ==================================================

def test_split(split_name):

    print("\n" + "=" * 60)
    print(
        f"TESTING {split_name.upper()} DATASET"
    )
    print("=" * 60)

    dataset = COD10KDataset(
        dataset_dir=DATASET_DIR,
        split=split_name,
        image_size=352
    )

    print(
        f"\nNumber of samples: "
        f"{len(dataset)}"
    )

    image, mask = dataset[0]

    print(
        f"Image shape: "
        f"{image.shape}"
    )

    print(
        f"Mask shape: "
        f"{mask.shape}"
    )

    print(
        f"Image dtype: "
        f"{image.dtype}"
    )

    print(
        f"Mask dtype: "
        f"{mask.dtype}"
    )

    print(
        f"Image min: "
        f"{image.min():.4f}"
    )

    print(
        f"Image max: "
        f"{image.max():.4f}"
    )

    print(
        f"Mask unique values: "
        f"{torch.unique(mask)}"
    )


def main():

    print("=" * 60)
    print("COD10K PYTORCH DATASET TEST")
    print("=" * 60)

    test_split("train")

    test_split("val")

    test_split("test")

    print("\n" + "=" * 60)
    print("PYTORCH DATASET TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()