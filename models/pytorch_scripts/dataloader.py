from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset

from cod10k_dataset import COD10KDataset


# ==================================================
# CONFIGURATION
# ==================================================

IMAGE_SIZE = 352
BATCH_SIZE = 4

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()


# ==================================================
# PATHS
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = CURRENT_DIR.parent

# Valid dataset names
VALID_DATASETS = ["COD10K_CAM", "CAMO", "NC4K"]


# ==================================================
# CREATE DATALOADERS
# ==================================================

def get_dataloaders(dataset_name="COD10K_CAM"):
    """
    Create dataloaders for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset to use.
                           Valid options: "COD10K_CAM", "CAMO", "NC4K"

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset_name: {dataset_name}. "
                         f"Valid options are: {VALID_DATASETS}")

    DATASET_DIR = (
        PROJECT_DIR
        / "datasets"
        / dataset_name
    )

    train_dataset = COD10KDataset(
        dataset_dir=DATASET_DIR,
        split="train",
        image_size=IMAGE_SIZE
    )

    val_dataset = COD10KDataset(
        dataset_dir=DATASET_DIR,
        split="val",
        image_size=IMAGE_SIZE
    )

    test_dataset = COD10KDataset(
        dataset_dir=DATASET_DIR,
        split="test",
        image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


def get_combined_dataloaders(dataset_names):
    """
    Create combined dataloaders for multiple datasets.

    Args:
        dataset_names (list): List of dataset names to combine.
                           Valid options: ["COD10K_CAM", "CAMO", "NC4K"]

    Returns:
        tuple: (combined_train_loader, combined_val_loader, test_loaders)
               where test_loaders is a dict mapping dataset_name to test DataLoader
    """
    # Validate dataset names
    for name in dataset_names:
        if name not in VALID_DATASETS:
            raise ValueError(f"Invalid dataset_name: {name}. "
                             f"Valid options are: {VALID_DATASETS}")

    # Create individual dataset objects for each split
    train_datasets = []
    val_datasets = []
    test_datasets = {}

    for dataset_name in dataset_names:
        DATASET_DIR = (
            PROJECT_DIR
            / "datasets"
            / dataset_name
        )

        # Create train dataset
        train_dataset = COD10KDataset(
            dataset_dir=DATASET_DIR,
            split="train",
            image_size=IMAGE_SIZE
        )
        train_datasets.append(train_dataset)

        # Create validation dataset
        val_dataset = COD10KDataset(
            dataset_dir=DATASET_DIR,
            split="val",
            image_size=IMAGE_SIZE
        )
        val_datasets.append(val_dataset)

        # Create test dataset (kept separate)
        test_dataset = COD10KDataset(
            dataset_dir=DATASET_DIR,
            split="test",
            image_size=IMAGE_SIZE
        )
        test_datasets[dataset_name] = test_dataset

    # Combine training datasets
    combined_train_dataset = ConcatDataset(train_datasets)

    # Combine validation datasets
    combined_val_dataset = ConcatDataset(val_datasets)

    # Create DataLoaders
    combined_train_loader = DataLoader(
        combined_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    combined_val_loader = DataLoader(
        combined_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Create separate test DataLoaders
    test_loaders = {}
    for dataset_name, test_dataset in test_datasets.items():
        test_loaders[dataset_name] = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )

    return (
        combined_train_loader,
        combined_val_loader,
        test_loaders
    )