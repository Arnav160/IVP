
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_DIR = Path(
    r"C:\Users\Arnav Agrawal\Desktop\college\IVP_projevt\project"
)
DEPTH_ANYTHING_DIR = Path(
    r"C:\Users\Arnav Agrawal\Desktop\college\IVP_projevt\Depth-Anything-V2"
)
CHECKPOINT = (
    DEPTH_ANYTHING_DIR / "checkpoints" / "depth_anything_v2_vits.pth"
)

DATASETS = ["COD10K_CAM", "CAMO", "NC4K"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Save depth maps separately from the original images and masks.
# Output: project/depth_maps/DATASET/SPLIT/filename.npy
OUTPUT_ROOT = PROJECT_DIR / "depth_maps"

# --------------------------------------------------
# Model setup
# --------------------------------------------------
sys.path.insert(0, str(DEPTH_ANYTHING_DIR))
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}

model = DepthAnythingV2(**model_configs["vits"])
state_dict = torch.load(
    CHECKPOINT, map_location="cpu", weights_only=True
)
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

# --------------------------------------------------
# Generate cached relative-depth maps
# --------------------------------------------------
total_images = 0
skipped_images = 0
failed_images = []

for dataset in DATASETS:
    for split in SPLITS:
        image_dir = PROJECT_DIR / "datasets" / dataset / split / "images"

        if not image_dir.exists():
            print(f"Skipping missing folder: {image_dir}")
            continue

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        output_dir = OUTPUT_ROOT / dataset / split
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{dataset}/{split}: {len(images)} images")

        for image_path in tqdm(images, desc=f"{dataset}/{split}"):
            output_path = output_dir / f"{image_path.stem}.npy"

            # Allow safe restart without recalculating completed files.
            if output_path.exists() and output_path.stat().st_size > 0:
                skipped_images += 1
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                failed_images.append(str(image_path))
                continue

            try:
                with torch.inference_mode():
                    depth = model.infer_image(image)

                depth = np.asarray(depth, dtype=np.float32)

                if depth.ndim != 2 or not np.isfinite(depth).all():
                    raise ValueError("Invalid depth output")

                # Preserve native prediction resolution and relative values.
                np.save(output_path, depth)
                total_images += 1

            except Exception as exc:
                failed_images.append(f"{image_path}: {exc}")

        print("Saved depth maps in:", output_dir)

print("\n========== SUMMARY ==========")
print("New depth maps saved:", total_images)
print("Existing maps skipped:", skipped_images)
print("Failed images:", len(failed_images))
print("Output folder:", OUTPUT_ROOT)

if failed_images:
    failure_log = OUTPUT_ROOT / "failed_images.txt"
    failure_log.write_text("\n".join(failed_images), encoding="utf-8")
    print("Failure details:", failure_log)
