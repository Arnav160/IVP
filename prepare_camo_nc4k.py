#!/usr/bin/env python3
"""
Prepare CAMO and NC4K datasets for PyTorch camouflaged object detection/segmentation.

This script:
- Creates working copies of CAMO and NC4K datasets under project/datasets/
- Does NOT modify the original datasets
- Verifies image-mask correspondence before copying
- For CAMO: splits official training set (1000) into 800 train/200 val (seed=42), keeps official test (250) as test
- For CAMO: converts GT masks to binary (0/255) based on inspection of pixel values
- For NC4K: splits all images into 80% train/10% val/10% test (seed=42)
- Copies files preserving original structure
- Prints detailed summary and verification
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Project root (directory containing this script)
PROJECT_DIR = Path(__file__).resolve().parent

# Original datasets
CAMO_ORIGINAL = PROJECT_DIR.parent / 'CAMO-V.1.0-CVIU2019'
NC4K_ORIGINAL = PROJECT_DIR.parent / 'NC4K'

# Working datasets
WORKING_DATASET_DIR = PROJECT_DIR / 'datasets'
CAMO_WORKING = WORKING_DATASET_DIR / 'CAMO'
NC4K_WORKING = WORKING_DATASET_DIR / 'NC4K'

# Verify original datasets exist
if not CAMO_ORIGINAL.exists():
    raise FileNotFoundError(f"Original CAMO dataset not found at: {CAMO_ORIGINAL}")
if not NC4K_ORIGINAL.exists():
    raise FileNotFoundError(f"Original NC4K dataset not found at: {NC4K_ORIGINAL}")

# Helper functions
def get_image_stems(image_dir):
    """Return set of image stems (filename without extension) for .jpg files in directory."""
    stems = set()
    if image_dir.is_dir():
        for f in image_dir.glob('*.jpg'):
            stems.add(f.stem)
    return stems

def get_mask_stems(mask_dir):
    """Return set of mask stems (filename without extension) for .png files in directory."""
    stems = set()
    if mask_dir.is_dir():
        for f in mask_dir.glob('*.png'):
            stems.add(f.stem)
    return stems

def verify_dataset_correspondence(image_dirs, mask_dir, dataset_name):
    """
    Verify that the union of image stems from given directories matches mask stems.
    image_dirs: list of Path objects containing images.
    Returns True if perfect match.
    """
    all_image_stems = set()
    for img_dir in image_dirs:
        all_image_stems.update(get_image_stems(img_dir))
    mask_stems = get_mask_stems(mask_dir)

    images_without_masks = all_image_stems - mask_stems
    masks_without_images = mask_stems - all_image_stems

    print(f"  [{dataset_name}] Total images: {len(all_image_stems)}, Total masks: {len(mask_stems)}")
    print(f"    Images without masks: {len(images_without_masks)}")
    print(f"    Masks without images: {len(masks_without_images)}")

    if len(images_without_masks) > 0 or len(masks_without_images) > 0:
        print(f"    WARNING: Mismatch found in {dataset_name}!")
        if images_without_masks:
            print(f"    Example images without masks: {sorted(list(images_without_masks))[:3]}")
        if masks_without_images:
            print(f"    Example masks without images: {sorted(list(masks_without_images))[:3]}")
        return False
    return True

def verify_split_correspondence(image_dir, mask_dir, dataset_name, split_name):
    """
    Verify that for each image in image_dir, there is a mask with same stem in mask_dir.
    Does NOT complain about extra masks in mask_dir (they may belong to other splits).
    """
    image_stems = get_image_stems(image_dir)
    mask_stems = get_mask_stems(mask_dir)

    images_without_masks = image_stems - mask_stems
    # We ignore masks_without_images for split-specific verification

    print(f"  [{dataset_name} {split_name}] Images: {len(image_stems)}")
    print(f"    Images without masks: {len(images_without_masks)}")

    if len(images_without_masks) > 0:
        print(f"    WARNING: Some images lack masks in {dataset_name} {split_name}!")
        print(f"    Example images without masks: {sorted(list(images_without_masks))[:3]}")
        return False
    return True

def copy_and_process_images(image_src_dir, image_dst_dir, image_stems_to_copy):
    """Copy images from source to destination directory."""
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for stem in image_stems_to_copy:
        src_file = image_src_dir / f"{stem}.jpg"
        dst_file = image_dst_dir / f"{stem}.jpg"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            copied += 1
        else:
            print(f"    WARNING: Source image not found: {src_file}")
    return copied

def copy_and_process_masks_camo(mask_src_dir, mask_dst_dir, mask_stems_to_copy):
    """
    Copy and process CAMO masks: convert to binary (0/255).
    We assume non-zero pixels are foreground (object) and zero is background.
    """
    mask_dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for stem in mask_stems_to_copy:
        src_file = mask_src_dir / f"{stem}.png"
        dst_file = mask_dst_dir / f"{stem}.png"
        if src_file.exists():
            # Open mask
            mask = Image.open(src_file)
            # Convert to numpy array
            mask_array = np.array(mask)
            # Determine unique values
            unique_vals = np.unique(mask_array)
            # Convert to binary: foreground (non-zero) -> 255, background (zero) -> 0
            binary_mask = np.where(mask_array > 0, 255, 0).astype(np.uint8)
            # Save as PNG
            mask_binary = Image.fromarray(binary_mask)
            mask_binary.save(dst_file)
            copied += 1
            # Print conversion info for first few masks
            if copied <= 3:
                print(f"    Mask {stem}: unique values before={unique_vals}, after=[0, 255]")
        else:
            print(f"    WARNING: Source mask not found: {src_file}")
    return copied

def copy_and_process_masks_nc4k(mask_src_dir, mask_dst_dir, mask_stems_to_copy):
    """Copy NC4K masks as-is (they are already binary 0/255)."""
    mask_dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for stem in mask_stems_to_copy:
        src_file = mask_src_dir / f"{stem}.png"
        dst_file = mask_dst_dir / f"{stem}.png"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            copied += 1
        else:
            print(f"    WARNING: Source mask not found: {src_file}")
    return copied

def verify_split(image_dir, mask_dir, dataset_name, split_name, check_binary=False):
    """Verify a split: count files, check correspondence (images have masks), and optionally check mask binary values."""
    image_files = list(image_dir.glob('*.jpg')) if image_dir.is_dir() else []
    mask_files = list(mask_dir.glob('*.png')) if mask_dir.is_dir() else []

    print(f"\n  Verification for {dataset_name} {split_name}:")
    print(f"    Image files: {len(image_files)}")
    print(f"    Mask files:  {len(mask_files)}")

    # Check that each image has a mask (we don't care about extra masks)
    image_stems = {f.stem for f in image_files}
    mask_stems = {f.stem for f in mask_files}
    images_without_masks = image_stems - mask_stems

    print(f"    Images without masks: {len(images_without_masks)}")

    # Check for non-image/mask files
    non_image = [f for f in image_dir.iterdir() if f.is_file() and not f.name.lower().endswith('.jpg')] if image_dir.is_dir() else []
    non_mask = [f for f in mask_dir.iterdir() if f.is_file() and not f.name.lower().endswith('.png')] if mask_dir.is_dir() else []
    print(f"    Non-image files: {len(non_image)}")
    print(f"    Non-mask files:  {len(non_mask)}")

    # Randomly select 5 pairs for detailed verification
    if len(image_files) >= 5:
        sample_stems = random.sample(list(image_stems), min(5, len(image_stems)))
    else:
        sample_stems = list(image_stems)

    all_good = True
    for stem in sample_stems:
        img_path = image_dir / f"{stem}.jpg"
        mask_path = mask_dir / f"{stem}.png"
        if not img_path.exists() or not mask_path.exists():
            print(f"    ERROR: Missing file for stem {stem}")
            all_good = False
            continue
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            with Image.open(mask_path) as mask:
                mask_width, mask_height = mask.size
                if img_width != mask_width or img_height != mask_height:
                    print(f"    ERROR: Size mismatch for {stem}: image ({img_width}x{img_height}) vs mask ({mask_width}x{mask_height})")
                    all_good = False
                # Check mask binary values if requested
                if check_binary:
                    mask_array = np.array(mask)
                    unique_vals = np.unique(mask_array)
                    if not set(unique_vals).issubset({0, 255}):
                        print(f"    ERROR: Mask {stem} contains non-binary values: {unique_vals}")
                        all_good = False
                    else:
                        # Count foreground pixels
                        fg_pixels = np.sum(mask_array == 255)
                        bg_pixels = np.sum(mask_array == 0)
                        print(f"    Mask {stem}: shape={mask_array.shape}, foreground pixels={fg_pixels}, background pixels={bg_pixels}")
        except Exception as e:
            print(f"    ERROR: Could not read files for stem {stem}: {e}")
            all_good = False

    if all_good and len(sample_stems) > 0:
        print(f"    All {len(sample_stems)} sampled pairs verified successfully.")
    return all_good and len(images_without_masks) == 0

def main():
    print("=" * 70)
    print("PREPARING CAMO AND NC4K DATASETS FOR PYTORCH")
    print("=" * 70)
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    # ============================================================
    # PREPARE CAMO DATASET
    # ============================================================
    print("PREPARING CAMO DATASET")
    print("-" * 70)

    # Original CAMO paths
    camino_images_train_original = CAMO_ORIGINAL / 'Images' / 'Train'
    camino_images_test_original = CAMO_ORIGINAL / 'Images' / 'Test'
    camino_masks_original = CAMO_ORIGINAL / 'GT'

    # Verify original CAMO dataset as a whole (all images vs all masks)
    print("Verifying original CAMO dataset (all images vs all masks)...")
    camino_all_image_dirs = [camino_images_train_original, camino_images_test_original]
    whole_ok = verify_dataset_correspondence(camino_all_image_dirs, camino_masks_original, "CAMO")
    if not whole_ok:
        print("ERROR: Original CAMO dataset has mismatches between total images and masks. Aborting.")
        return

    # Get stems for each split
    camino_train_stems = get_image_stems(camino_images_train_original)
    camino_test_stems = get_image_stems(camino_images_test_original)

    print(f"  Original CAMO Training images: {len(camino_train_stems)}")
    print(f"  Original CAMO Test images:     {len(camino_test_stems)}")
    print(f"  Original CAMO GT masks:        {len(get_mask_stems(camino_masks_original))}")

    # Verify that training and test are disjoint and union equals total
    if not camino_train_stems.isdisjoint(camino_test_stems):
        print("ERROR: CAMO training and test sets overlap. Aborting.")
        return
    if camino_train_stems.union(camino_test_stems) != set().union(*[get_image_stems(d) for d in camino_all_image_dirs]):
        print("ERROR: Union of training and test does not equal total images. Aborting.")
        return

    print(f"\nSplit verification:")
    print(f"  Training and Test are disjoint: {camino_train_stems.isdisjoint(camino_test_stems)}")
    print(f"  Training + Test = Total: {len(camino_train_stems) + len(camino_test_stems) == len(camino_train_stems.union(camino_test_stems))}")

    # Split CAMO training set: 800 train, 200 val (from 1000 total)
    camino_train_list = list(camino_train_stems)
    random.shuffle(camino_train_list)
    camino_train_split = set(camino_train_list[:800])
    camino_val_split = set(camino_train_list[800:])

    print(f"\nSplit CAMO training set (seed={RANDOM_SEED}):")
    print(f"  Training:  {len(camino_train_split)} images")
    print(f"  Validation: {len(camino_val_split)} images")
    print(f"  Test (official): {len(camino_test_stems)} images")

    # Create CAMO working directory structure
    print("\nCreating CAMO working dataset...")
    shutil.rmtree(CAMO_WORKING, ignore_errors=True)  # Start fresh

    # Define split directories
    splits = [
        ('train', camino_train_split, camino_images_train_original),
        ('val',   camino_val_split,   camino_images_train_original),
        ('test',  camino_test_stems,  camino_images_test_original)
    ]

    for split_name, stems_to_copy, images_src_dir in splits:
        print(f"\n  Processing CAMO {split_name} split...")
        # Verify split correspondence (each image has a mask in GT)
        if not verify_split_correspondence(images_src_dir, camino_masks_original, "CAMO", split_name):
            print(f"ERROR: CAMO {split_name} split has images without masks. Aborting.")
            return
        # Images
        img_dst_dir = CAMO_WORKING / split_name / 'images'
        img_copied = copy_and_process_images(images_src_dir, img_dst_dir, stems_to_copy)
        # Masks (with conversion)
        mask_dst_dir = CAMO_WORKING / split_name / 'masks'
        mask_copied = copy_and_process_masks_camo(camino_masks_original, mask_dst_dir, stems_to_copy)
        print(f"    Copied {img_copied} images and {mask_copied} masks.")
        # Verify correspondence in working set (exact match)
        img_stems_working = {f.stem for f in img_dst_dir.glob('*.jpg')}
        mask_stems_working = {f.stem for f in mask_dst_dir.glob('*.png')}
        if img_stems_working != mask_stems_working:
            print(f"ERROR: CAMO {split_name} split has mismatched image-mask sets in working directory.")
            print(f"  Images without masks: {len(img_stems_working - mask_stems_working)}")
            print(f"  Masks without images: {len(mask_stems_working - img_stems_working)}")
            return

    # ============================================================
    # PREPARE NC4K DATASET
    # ============================================================
    print("\n\nPREPARING NC4K DATASET")
    print("-" * 70)

    # Original NC4K paths
    nc4k_images_original = NC4K_ORIGINAL / 'Imgs'
    nc4k_masks_original = NC4K_ORIGINAL / 'GT'

    # Verify original NC4K dataset
    print("Verifying original NC4K dataset...")
    nc4k_ok = verify_dataset_correspondence([nc4k_images_original], nc4k_masks_original, "NC4K")
    if not nc4k_ok:
        print("ERROR: Original NC4K dataset has mismatches. Aborting.")
        return

    nc4k_all_stems = get_image_stems(nc4k_images_original)
    nc4k_mask_stems = get_mask_stems(nc4k_masks_original)
    print(f"  Original NC4K images: {len(nc4k_all_stems)}")
    print(f"  Original NC4K GT masks: {len(nc4k_mask_stems)}")

    # Split NC4K: 80% train, 10% val, 10% test
    nc4k_list = list(nc4k_all_stems)
    random.shuffle(nc4k_list)
    n_total = len(nc4k_list)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val  # remainder

    nc4k_train_split = set(nc4k_list[:n_train])
    nc4k_val_split   = set(nc4k_list[n_train:n_train + n_val])
    nc4k_test_split  = set(nc4k_list[n_train + n_val:])

    print(f"\nSplit NC4K dataset (seed={RANDOM_SEED}, total={n_total}):")
    print(f"  Training: {len(nc4k_train_split)} images ({n_train}/{n_total} = {n_train/n_total*100:.1f}%)")
    print(f"  Validation: {len(nc4k_val_split)} images ({n_val}/{n_total} = {n_val/n_total*100:.1f}%)")
    print(f"  Test: {len(nc4k_test_split)} images ({n_test}/{n_total} = {n_test/n_total*100:.1f}%)")

    # Create NC4K working directory structure
    print("\nCreating NC4K working dataset...")
    shutil.rmtree(NC4K_WORKING, ignore_errors=True)  # Start fresh

    # Define split directories
    nc4k_splits = [
        ('train', nc4k_train_split, nc4k_images_original),
        ('val',   nc4k_val_split,   nc4k_images_original),
        ('test',  nc4k_test_split,  nc4k_images_original)
    ]

    for split_name, stems_to_copy, images_src_dir in nc4k_splits:
        print(f"\n  Processing NC4K {split_name} split...")
        # Verify split correspondence
        if not verify_split_correspondence(images_src_dir, nc4k_masks_original, "NC4K", split_name):
            print(f"ERROR: NC4K {split_name} split has images without masks. Aborting.")
            return
        # Images
        img_dst_dir = NC4K_WORKING / split_name / 'images'
        img_copied = copy_and_process_images(images_src_dir, img_dst_dir, stems_to_copy)
        # Masks (as-is, but we'll verify they are binary)
        mask_dst_dir = NC4K_WORKING / split_name / 'masks'
        mask_copied = copy_and_process_masks_nc4k(nc4k_masks_original, mask_dst_dir, stems_to_copy)
        print(f"    Copied {img_copied} images and {mask_copied} masks.")
        # Verify correspondence in working set (exact match)
        img_stems_working = {f.stem for f in img_dst_dir.glob('*.jpg')}
        mask_stems_working = {f.stem for f in mask_dst_dir.glob('*.png')}
        if img_stems_working != mask_stems_working:
            print(f"ERROR: NC4K {split_name} split has mismatched image-mask sets in working directory.")
            print(f"  Images without masks: {len(img_stems_working - mask_stems_working)}")
            print(f"  Masks without images: {len(mask_stems_working - img_stems_working)}")
            return

    # ============================================================
    # FINAL VERIFICATION
    # ============================================================
    print("\n\n" + "=" * 70)
    print("FINAL VERIFICATION OF WORKING DATASETS")
    print("=" * 70)

    all_verification_passed = True

    # Verify CAMO splits
    for split_name in ['train', 'val', 'test']:
        img_dir = CAMO_WORKING / split_name / 'images'
        mask_dir = CAMO_WORKING / split_name / 'masks'
        passed = verify_split(img_dir, mask_dir, "CAMO", split_name, check_binary=True)
        all_verification_passed = all_verification_passed and passed

    # Verify NC4K splits
    for split_name in ['train', 'val', 'test']:
        img_dir = NC4K_WORKING / split_name / 'images'
        mask_dir = NC4K_WORKING / split_name / 'masks'
        passed = verify_split(img_dir, mask_dir, "NC4K", split_name, check_binary=True)
        all_verification_passed = all_verification_passed and passed

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    def count_split(dataset_working, split_name):
        img_dir = dataset_working / split_name / 'images'
        mask_dir = dataset_working / split_name / 'masks'
        img_count = len(list(img_dir.glob('*.jpg'))) if img_dir.is_dir() else 0
        mask_count = len(list(mask_dir.glob('*.png'))) if mask_dir.is_dir() else 0
        return img_count, mask_count

    print("CAMO WORKING DATASET")
    for split in ['train', 'val', 'test']:
        img_count, mask_count = count_split(CAMO_WORKING, split)
        print(f"{split.capitalize():>6}: {img_count:>4} images / {mask_count:>4} masks")

    print("\nNC4K WORKING DATASET")
    for split in ['train', 'val', 'test']:
        img_count, mask_count = count_split(NC4K_WORKING, split)
        print(f"{split.capitalize():>6}: {img_count:>4} images / {mask_count:>4} masks")

    print()
    print(f"Random seed used: {RANDOM_SEED}")
    print(f"Original CAMO dataset preserved at: {CAMO_ORIGINAL}")
    print(f"Original NC4K dataset preserved at: {NC4K_ORIGINAL}")
    print(f"COD10K_CAM dataset (if exists) preserved at: {PROJECT_DIR / 'datasets' / 'COD10K_CAM'}")
    print()
    if all_verification_passed:
        print("[PASS] ALL VERIFICATIONS PASSED: Datasets prepared successfully.")
        print("[PASS] Image-mask pairs are valid.")
        print("[PASS] CAMO masks converted to binary (0/255).")
        print("[PASS] NC4K masks verified as binary (0/255).")
    else:
        print("[FAIL] SOME VERIFICATIONS FAILED: Please check the output above.")

    print("=" * 70)

if __name__ == "__main__":
    main()