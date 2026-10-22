#!/usr/bin/env python3
"""
Test script for get_combined_dataloaders function.
"""

import torch
from pathlib import Path
import sys

# Add project root to path so we can import from pytorch
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR / "pytorch"))

from dataloader import get_combined_dataloaders, get_dataloaders

def test_combined_dataloader():
    """Test the combined dataloader function."""
    print("Testing get_combined_dataloaders...")

    # Test with all three datasets
    dataset_names = ["COD10K_CAM", "CAMO", "NC4K"]
    combined_train_loader, combined_val_loader, test_loaders = get_combined_dataloaders(dataset_names)

    # Check combined train sample count
    expected_train_count = 2736 + 800 + 3296  # COD10K_CAM train + CAMO train + NC4K train
    actual_train_count = len(combined_train_loader.dataset)
    print(f"Combined train samples: {actual_train_count} (expected: {expected_train_count})")
    assert actual_train_count == expected_train_count, f"Expected {expected_train_count}, got {actual_train_count}"

    # Check combined validation sample count
    expected_val_count = 304 + 200 + 412  # COD10K_CAM val + CAMO val + NC4K val
    actual_val_count = len(combined_val_loader.dataset)
    print(f"Combined validation samples: {actual_val_count} (expected: {expected_val_count})")
    assert actual_val_count == expected_val_count, f"Expected {expected_val_count}, got {actual_val_count}"

    # Check individual test dataset counts
    expected_cod10k_test = 2026
    expected_camo_test = 250
    expected_nc4k_test = 413

    actual_cod10k_test = len(test_loaders["COD10K_CAM"].dataset)
    actual_camo_test = len(test_loaders["CAMO"].dataset)
    actual_nc4k_test = len(test_loaders["NC4K"].dataset)

    print(f"COD10K_CAM test samples: {actual_cod10k_test} (expected: {expected_cod10k_test})")
    print(f"CAMO test samples: {actual_camo_test} (expected: {expected_camo_test})")
    print(f"NC4K test samples: {actual_nc4k_test} (expected: {expected_nc4k_test})")

    assert actual_cod10k_test == expected_cod10k_test, f"Expected {expected_cod10k_test}, got {actual_cod10k_test}"
    assert actual_camo_test == expected_camo_test, f"Expected {expected_camo_test}, got {actual_camo_test}"
    assert actual_nc4k_test == expected_nc4k_test, f"Expected {expected_nc4k_test}, got {actual_nc4k_test}"

    # Check batch shapes and types
    print("\nChecking batch properties...")
    for images, masks in combined_train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch mask shape: {masks.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Mask dtype: {masks.dtype}")

        # Check shapes
        assert images.shape == (4, 3, 352, 352), f"Expected image shape [4, 3, 352, 352], got {images.shape}"
        assert masks.shape == (4, 1, 352, 352), f"Expected mask shape [4, 1, 352, 352], got {masks.shape}"

        # Check dtypes
        assert images.dtype == torch.float32, f"Expected image dtype torch.float32, got {images.dtype}"
        assert masks.dtype == torch.float32, f"Expected mask dtype torch.float32, got {masks.dtype}"

        # Check mask values are 0.0 and 1.0
        unique_vals = torch.unique(masks)
        print(f"Unique mask values in batch: {unique_vals}")
        assert set(unique_vals.tolist()) <= {0.0, 1.0}, f"Mask values should be 0.0 and 1.0, got {unique_vals.tolist()}"

        # Check CUDA transfer if available
        if torch.cuda.is_available():
            images_cuda = images.cuda()
            masks_cuda = masks.cuda()
            print(f"CUDA transfer successful: images on {images_cuda.device}, masks on {masks_cuda.device}")
            assert images_cuda.is_cuda, "Images should be on CUDA"
            assert masks_cuda.is_cuda, "Masks should be on CUDA"
        else:
            print("CUDA not available, skipping CUDA transfer check")

        # Only check first batch
        break

    # Test that existing get_dataloaders still works
    print("\nTesting existing get_dataloaders function...")
    for dataset_name in ["COD10K_CAM", "CAMO", "NC4K"]:
        train_loader, val_loader, test_loader = get_dataloaders(dataset_name)
        print(f"{dataset_name} - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        # Quick check that we can get a batch
        for images, masks in train_loader:
            assert images.shape == (4, 3, 352, 352)
            assert masks.shape == (4, 1, 352, 352)
            break

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_combined_dataloader()