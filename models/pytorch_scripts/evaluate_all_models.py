import sys
import csv
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

sys.path.insert(0, str(PROJECT_DIR / "pytorch"))

from combined_dataset import CombinedDataset
from depth_dataset import RGBDepthDataset


def load_model(folder, checkpoint_name, device):
    model_path = PROJECT_DIR / "models" / folder / "model.py"

    spec = importlib.util.spec_from_file_location(
        f"model_{folder}", model_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Canny checkpoint uses 4 channels; RGB baseline uses 3.
    if checkpoint_name == "unet_resnet34_canny_extended_best.pth":
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=4,
            classes=1,
            activation=None,
        )
    else:
        model = module.create_model()

    model = model.to(device)

    checkpoint = torch.load(
        CHECKPOINT_DIR / checkpoint_name,
        map_location=device,
        weights_only=True,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded {folder}: {checkpoint_name}")
    return model


def calculate_metrics(predictions, targets, smooth=1e-7):
    predictions = predictions.bool()
    targets = targets.bool()

    intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
    pred_count = predictions.sum(dim=(1, 2, 3)).float()
    target_count = targets.sum(dim=(1, 2, 3)).float()

    dice = (2 * intersection + smooth) / (
        pred_count + target_count + smooth
    )
    iou = (intersection + smooth) / (
        pred_count + target_count - intersection + smooth
    )

    return dice, iou


def evaluate_model(model, loader, device):
    dice_sum = 0.0
    iou_sum = 0.0
    image_count = 0

    global_intersection = 0
    global_pred = 0
    global_target = 0

    with torch.no_grad():
        for inputs, masks in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            masks = masks.to(device)

            logits = model(inputs)
            predictions = torch.sigmoid(logits) >= 0.5
            targets = masks >= 0.5

            dice, iou = calculate_metrics(predictions, targets)

            dice_sum += dice.sum().item()
            iou_sum += iou.sum().item()
            image_count += inputs.size(0)

            global_intersection += (predictions & targets).sum().item()
            global_pred += predictions.sum().item()
            global_target += targets.sum().item()

    mean_dice = dice_sum / image_count
    mean_iou = iou_sum / image_count

    global_dice = 2 * global_intersection / max(
        global_pred + global_target, 1
    )
    global_iou = global_intersection / max(
        global_pred + global_target - global_intersection, 1
    )

    return mean_dice, mean_iou, global_dice, global_iou


class RGBOnlyView(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        inputs, mask = self.base_dataset[index]
        return inputs[:3], mask


class RGBAndCannyView(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        inputs, mask = self.base_dataset[index]
        # RGB channels plus Canny channel; omit depth.
        return torch.cat([inputs[:3], inputs[3:4]], dim=0), mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_configs = [
        (
            "RGB",
            "unet_resnet",
            "unet_resnet34_all_datasets_best.pth",
        ),
        (
            "RGB+Canny",
            "unet_resnet",
            "unet_resnet34_canny_extended_best.pth",
        ),
        (
            "RGB+Depth",
            "unet_resnet_depth",
            "unet_resnet34_depth_best.pth",
        ),
        (
            "RGB+Canny+Depth",
            "unet_resnet_combined",
            "unet_resnet34_rgb_canny_depth_best.pth",
        ),
    ]

    models = {}
    for label, folder, checkpoint_name in model_configs:
        models[label] = load_model(folder, checkpoint_name, device)

    dataset_names = ("COD10K_CAM", "CAMO", "NC4K")
    all_results = []

    for dataset_name in dataset_names:
        print(f"\n========== {dataset_name} ==========")

        depth_dataset = RGBDepthDataset(
            project_dir=PROJECT_DIR,
            datasets=(dataset_name,),
            split="test",
            image_size=352,
        )

        combined_dataset = CombinedDataset(
            project_dir=PROJECT_DIR,
            datasets=(dataset_name,),
            split="test",
            image_size=352,
        )

        loaders = {
            "RGB": DataLoader(
                RGBOnlyView(depth_dataset),
                batch_size=4,
                shuffle=False,
                num_workers=0,
            ),
            "RGB+Canny": DataLoader(
                RGBAndCannyView(combined_dataset),
                batch_size=4,
                shuffle=False,
                num_workers=0,
            ),
            "RGB+Depth": DataLoader(
                depth_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
            ),
            "RGB+Canny+Depth": DataLoader(
                combined_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
            ),
        }

        for label, _, _ in model_configs:
            metrics = evaluate_model(models[label], loaders[label], device)

            row = (
                dataset_name,
                label,
                len(loaders[label].dataset),
                *metrics,
            )
            all_results.append(row)

            print(
                f"{label:<18} "
                f"Mean Dice={metrics[0]:.6f}  "
                f"Mean IoU={metrics[1]:.6f}  "
                f"Global Dice={metrics[2]:.6f}  "
                f"Global IoU={metrics[3]:.6f}"
            )

    output_csv = CHECKPOINT_DIR / "all_models_standardized_results.csv"

    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Dataset",
            "Model",
            "Images",
            "Mean Dice",
            "Mean IoU",
            "Global Dice",
            "Global IoU",
        ])
        writer.writerows(all_results)

    print("\nEvaluation complete!")
    print("Results saved to:", output_csv)


if __name__ == "__main__":
    main()