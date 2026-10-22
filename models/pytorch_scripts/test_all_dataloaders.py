#!/usr/bin/env python3
"""
Test script to verify that all three datasets (COD10K_CAM, CAMO, NC4K)
work correctly with the modified dataloader.
"""

import torch
from pathlib import Path
from dataloader import get_dataloaders

def test_dataset(dataset_name):
    """Test a single dataset and print results."""
    print(f"============================================================")
    print(f"{dataset_name} DATASET TEST")
    print(f"============================================================")

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(dataset_name=dataset_name)

    # Print dataset sizes
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Test one batch from each loader
    for loader_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        print(f"\n{loader_name} batch test:")

        # Get one batch
        batch = next(iter(loader))
        images, masks = batch

        # Print shapes and types
        print(f"  Image shape: {images.shape}")
        print(f"  Mask shape: {masks.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Mask dtype: {masks.dtype}")

        # Print value ranges
        print(f"  Image min/max: {images.min():.4f}/{images.max():.4f}")
        print(f"  Mask unique values: {torch.unique(masks)}")

        # Verify device
        print(f"  Device: {images.device}")

        # Test CUDA transfer if available
        if torch.cuda.is_available():
            images_cuda = images.to('cuda')
            masks_cuda = masks.to('cuda')
            print(f"  Successfully transferred to CUDA: {images_cuda.device}")
        else:
            print(f"  CUDA not available, skipping GPU transfer test")

    print()

def main():
    """Test all three datasets."""
    datasets = ["COD10K_CAM", "CAMO", "NC4K"]

    for dataset_name in datasets:
        try:
            test_dataset(dataset_name)
        except Exception as e:
            print(f"ERROR testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            print()

if __name__ == "__main__":
    main()