from pathlib import Path
import sys
import torch

# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent.parent

DATASET_DIR = PROJECT_DIR / "datasets"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
PYTORCH_DIR = PROJECT_DIR / "pytorch"

sys.path.append(str(PYTORCH_DIR))

# ============================================================
# IMPORTS
# ============================================================

try:
    from dataloader import get_combined_dataloaders
except Exception as e:
    print("\n❌ ERROR importing dataloader:")
    print(e)
    sys.exit(1)

try:
    from model import create_model
except Exception as e:
    print("\n❌ ERROR importing model:")
    print(e)
    sys.exit(1)

# ============================================================
# SETTINGS
# ============================================================

DATASETS = ["COD10K_CAM", "CAMO", "NC4K"]

EXPECTED_TRAIN = 6832
EXPECTED_VAL = 916

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("U-NET RESNET-34 COMBINED TRAINING READINESS CHECK")
print("=" * 70)

# ============================================================
# 1. PYTORCH / CUDA
# ============================================================

print("\n[1] PYTORCH / CUDA CHECK")
print("-" * 70)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device("cuda")
    print("✅ CUDA is available.")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ CUDA is NOT available.")
    print("Training can still run, but it will be much slower.")

# ============================================================
# 2. DATASET DIRECTORIES
# ============================================================

print("\n[2] DATASET DIRECTORY CHECK")
print("-" * 70)

all_dataset_dirs_ok = True

for dataset_name in DATASETS:
    dataset_path = DATASET_DIR / dataset_name

    print(f"\n{dataset_name}:")
    print(f"Path: {dataset_path}")

    if not dataset_path.exists():
        print("❌ Dataset directory NOT FOUND")
        all_dataset_dirs_ok = False
        continue

    print("✅ Dataset directory exists.")

    for split in ["train", "val", "test"]:
        images_dir = dataset_path / split / "images"
        masks_dir = dataset_path / split / "masks"

        if images_dir.exists():
            print(f"  ✅ {split}/images")
        else:
            print(f"  ❌ {split}/images MISSING")
            all_dataset_dirs_ok = False

        if masks_dir.exists():
            print(f"  ✅ {split}/masks")
        else:
            print(f"  ❌ {split}/masks MISSING")
            all_dataset_dirs_ok = False

# ============================================================
# 3. COMBINED DATALOADER
# ============================================================

print("\n[3] COMBINED DATALOADER CHECK")
print("-" * 70)

try:
    train_loader, val_loader, test_loaders = (
        get_combined_dataloaders(DATASETS)
    )

    print(f"Combined training samples: {len(train_loader.dataset)}")
    print(f"Combined validation samples: {len(val_loader.dataset)}")

    if len(train_loader.dataset) == EXPECTED_TRAIN:
        print("✅ Training sample count is correct.")
    else:
        print(
            f"❌ Training sample count is WRONG. "
            f"Expected {EXPECTED_TRAIN}."
        )

    if len(val_loader.dataset) == EXPECTED_VAL:
        print("✅ Validation sample count is correct.")
    else:
        print(
            f"❌ Validation sample count is WRONG. "
            f"Expected {EXPECTED_VAL}."
        )

except Exception as e:
    print("❌ Combined dataloader failed.")
    print(e)
    sys.exit(1)

# ============================================================
# 4. TEST DATASETS
# ============================================================

print("\n[4] TEST DATASET CHECK")
print("-" * 70)

expected_test_counts = {
    "COD10K_CAM": 2026,
    "CAMO": 250,
    "NC4K": 413,
}

for dataset_name, expected_count in expected_test_counts.items():

    if dataset_name not in test_loaders:
        print(f"❌ No test loader found for {dataset_name}")
        continue

    test_loader = test_loaders[dataset_name]
    actual_count = len(test_loader.dataset)

    print(
        f"{dataset_name}: "
        f"{actual_count} samples "
        f"(expected {expected_count})"
    )

    if actual_count == expected_count:
        print("  ✅ Correct")
    else:
        print("  ❌ Incorrect")

# ============================================================
# 5. BATCH / TENSOR CHECK
# ============================================================

print("\n[5] BATCH / TENSOR CHECK")
print("-" * 70)

try:
    images, masks = next(iter(train_loader))

    print(f"Image shape: {images.shape}")
    print(f"Mask shape:  {masks.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Mask dtype:  {masks.dtype}")

    print(
        f"Image min/max: "
        f"{images.min().item():.4f} / "
        f"{images.max().item():.4f}"
    )

    unique_masks = torch.unique(masks)

    print(f"Mask values: {unique_masks.tolist()}")

    if images.shape[1:] == (3, 352, 352):
        print("✅ Image shape is correct.")
    else:
        print("⚠️ Unexpected image shape.")

    if masks.shape[1:] == (1, 352, 352):
        print("✅ Mask shape is correct.")
    else:
        print("⚠️ Unexpected mask shape.")

    if images.dtype == torch.float32:
        print("✅ Image dtype is float32.")
    else:
        print("⚠️ Image dtype is not float32.")

    if masks.dtype == torch.float32:
        print("✅ Mask dtype is float32.")
    else:
        print("⚠️ Mask dtype is not float32.")

    if all(
        value.item() in [0.0, 1.0]
        for value in unique_masks
    ):
        print("✅ Masks are binary (0/1).")
    else:
        print("⚠️ Masks contain values other than 0/1.")

except Exception as e:
    print("❌ Could not read a training batch.")
    print(e)
    sys.exit(1)

# ============================================================
# 6. CUDA TRANSFER CHECK
# ============================================================

print("\n[6] CUDA TRANSFER CHECK")
print("-" * 70)

try:
    images_gpu = images.to(DEVICE)
    masks_gpu = masks.to(DEVICE)

    print(f"Images device: {images_gpu.device}")
    print(f"Masks device:  {masks_gpu.device}")

    if DEVICE.type == "cuda":
        if images_gpu.device.type == "cuda":
            print("✅ Images successfully transferred to GPU.")
        else:
            print("❌ Images did not transfer to GPU.")

        if masks_gpu.device.type == "cuda":
            print("✅ Masks successfully transferred to GPU.")
        else:
            print("❌ Masks did not transfer to GPU.")
    else:
        print("⚠️ Running on CPU, so GPU transfer was skipped.")

except Exception as e:
    print("❌ Device transfer failed.")
    print(e)

# ============================================================
# 7. MODEL CHECK
# ============================================================

print("\n[7] U-NET RESNET-34 MODEL CHECK")
print("-" * 70)

try:
    model = create_model().to(DEVICE)

    print("Model created successfully.")
    print(f"Model device: {next(model.parameters()).device}")

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Total parameters: {total_params:,}")

    print("✅ U-Net ResNet-34 model is ready.")

except Exception as e:
    print("❌ Model creation failed.")
    print(e)
    sys.exit(1)

# ============================================================
# 8. EXISTING CHECKPOINT CHECK
# ============================================================

print("\n[8] EXISTING CHECKPOINT CHECK")
print("-" * 70)

old_checkpoint = (
    CHECKPOINT_DIR /
    "unet_resnet34_best.pth"
)

combined_checkpoint = (
    CHECKPOINT_DIR /
    "unet_resnet34_combined_best.pth"
)

print(f"Original checkpoint:")
print(old_checkpoint)

if old_checkpoint.exists():
    print("✅ Original COD10K checkpoint exists.")
else:
    print("⚠️ Original checkpoint not found.")

print("\nCombined checkpoint:")
print(combined_checkpoint)

if combined_checkpoint.exists():
    print(
        "⚠️ Combined checkpoint already exists."
    )
    print(
        "Make sure you want to continue before "
        "overwriting it."
    )
else:
    print(
        "✅ No combined checkpoint exists yet."
    )
    print(
        "A new one can be created safely."
    )

# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("READINESS CHECK COMPLETED")
print("=" * 70)

print("\nTraining configuration:")
print(f"Datasets: {', '.join(DATASETS)}")
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nIf all major checks above show ✅:")
print("READY TO START COMBINED TRAINING.")
print("=" * 70)