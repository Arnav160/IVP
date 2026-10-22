
import sys
import random
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# --------------------------------------------------
# Paths and imports
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "pytorch"))

from depth_dataset import RGBDepthDataset


def import_model_factory(module_name, model_path):
    spec = spec_from_file_location(module_name, model_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_model


create_depth_model = import_model_factory(
    "depth_model_module",
    PROJECT_DIR / "models" / "unet_resnet_depth" / "model.py",
)
create_rgb_model = import_model_factory(
    "rgb_model_module",
    PROJECT_DIR / "models" / "unet_resnet" / "model.py",
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DEPTH_CHECKPOINT = (
    PROJECT_DIR / "checkpoints" / "unet_resnet34_depth_best.pth"
)
RGB_CHECKPOINT = (
    PROJECT_DIR / "checkpoints" / "unet_resnet34_all_datasets_best.pth"
)

DATASET_NAME = "CAMO"  # COD10K_CAM, CAMO, or NC4K
SPLIT = "test"
NUM_IMAGES = 6
IMAGE_SIZE = 352
THRESHOLD = 0.5
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, path):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def denormalize_rgb(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = tensor.cpu() * std + mean
    return rgb.clamp(0, 1).permute(1, 2, 0).numpy()


def dice_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    return (2 * intersection + 1e-7) / (denominator + 1e-7)


def make_error_overlay(rgb, prediction, target):
    """
    Green: correctly predicted foreground.
    Red: false-positive pixels.
    Blue: missed foreground pixels.
    """
    overlay = rgb.copy()

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = prediction & target
    false_positive = prediction & ~target
    false_negative = ~prediction & target

    # Blend colors with the original image.
    overlay[true_positive] = (
        0.45 * overlay[true_positive] + 0.55 * np.array([0.0, 1.0, 0.0])
    )
    overlay[false_positive] = (
        0.35 * overlay[false_positive] + 0.65 * np.array([1.0, 0.0, 0.0])
    )
    overlay[false_negative] = (
        0.35 * overlay[false_negative] + 0.65 * np.array([0.0, 0.3, 1.0])
    )

    return np.clip(overlay, 0, 1)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not DEPTH_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {DEPTH_CHECKPOINT}")
    if not RGB_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {RGB_CHECKPOINT}")

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_NAME} | Split: {SPLIT}")

    dataset = RGBDepthDataset(
        project_dir=PROJECT_DIR,
        datasets=(DATASET_NAME,),
        split=SPLIT,
        image_size=IMAGE_SIZE,
    )

    depth_model = create_depth_model().to(DEVICE)
    depth_model = load_checkpoint(depth_model, DEPTH_CHECKPOINT)

    rgb_model = create_rgb_model().to(DEVICE)
    rgb_model = load_checkpoint(rgb_model, RGB_CHECKPOINT)

    count = min(NUM_IMAGES, len(dataset))
    indices = random.sample(range(len(dataset)), count)

    # One sample per row, larger panels, clear spacing.
    columns = [
        "Input RGB",
        "Ground Truth",
        "RGB-only Prediction",
        "RGB + Depth Prediction",
        "RGB-only Errors",
        "RGB + Depth Errors",
        "Relative Depth Map",
    ]

    fig, axes = plt.subplots(
        count,
        len(columns),
        figsize=(21, 3.6 * count),
        squeeze=False,
    )

    for col, title in enumerate(columns):
        axes[0, col].set_title(
            title,
            fontsize=10,
            fontweight="bold",
            pad=12,
        )

    with torch.no_grad():
        for row, index in enumerate(indices):
            inputs, mask = dataset[index]

            rgb_batch = inputs[:3].unsqueeze(0).to(DEVICE)
            rgbd_batch = inputs.unsqueeze(0).to(DEVICE)

            rgb_logits = rgb_model(rgb_batch)
            depth_logits = depth_model(rgbd_batch)

            rgb_prediction = (
                torch.sigmoid(rgb_logits)[0, 0] >= THRESHOLD
            ).cpu().numpy()

            depth_prediction = (
                torch.sigmoid(depth_logits)[0, 0] >= THRESHOLD
            ).cpu().numpy()

            rgb = denormalize_rgb(inputs[:3])
            target = mask[0].numpy() >= 0.5
            depth_map = inputs[3].numpy()

            rgb_dice = dice_score(rgb_prediction, target)
            depth_dice = dice_score(depth_prediction, target)

            panels = [
                (rgb, None),
                (target, "gray"),
                (rgb_prediction, "gray"),
                (depth_prediction, "gray"),
                (make_error_overlay(rgb, rgb_prediction, target), None),
                (make_error_overlay(rgb, depth_prediction, target), None),
                (depth_map, "viridis"),
            ]

            for col, (image, cmap) in enumerate(panels):
                axes[row, col].imshow(
                    image,
                    cmap=cmap,
                    vmin=0 if cmap else None,
                    vmax=1 if cmap else None,
                )
                axes[row, col].axis("off")

            axes[row, 2].set_xlabel(
                f"Dice = {rgb_dice:.3f}", fontsize=9
            )
            axes[row, 3].set_xlabel(
                f"Dice = {depth_dice:.3f}", fontsize=9
            )
            axes[row, 0].text(
                -0.08, 0.5, f"Sample {row + 1}",
                transform=axes[row, 0].transAxes,
                rotation=90, va="center", ha="right",
                fontsize=9,
            )

            print(
                f"Sample {index}: RGB Dice={rgb_dice:.4f} | "
                f"RGB+Depth Dice={depth_dice:.4f} | "
                f"Difference={depth_dice - rgb_dice:+.4f}"
            )

    fig.suptitle(
        f"Camouflaged Object Segmentation Comparison — {DATASET_NAME}",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    # Legend for the error overlays.
    fig.text(
        0.5, 0.006,
        "Error overlays: Green = correctly detected foreground | "
        "Red = false positive | Blue = missed object pixels",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0.02, 0.025, 1, 0.975], w_pad=0.5, h_pad=1.2)

    output_dir = PROJECT_DIR / "visualizations" / "rgb_depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{DATASET_NAME}_{SPLIT}_comparison_clean.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved clean comparison to:\n{output_path}")

    plt.show()


if __name__ == "__main__":
    main()
