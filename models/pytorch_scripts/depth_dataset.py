
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class RGBDepthDataset(Dataset):
    """
    Loads RGB images, cached Depth Anything V2 depth maps,
    and ground-truth segmentation masks.
    """

    def __init__(
        self,
        project_dir,
        datasets=("COD10K_CAM", "CAMO", "NC4K"),
        split="train",
        image_size=352,
    ):
        self.project_dir = Path(project_dir)
        self.depth_root = self.project_dir / "depth_maps"
        self.image_size = image_size
        self.samples = []

        for dataset_name in datasets:
            image_dir = (
                self.project_dir / "datasets" / dataset_name
                / split / "images"
            )
            mask_dir = (
                self.project_dir / "datasets" / dataset_name
                / split / "masks"
            )
            depth_dir = (
                self.depth_root / dataset_name / split
            )

            if not image_dir.exists():
                raise FileNotFoundError(
                    f"Image folder not found: {image_dir}"
                )
            if not mask_dir.exists():
                raise FileNotFoundError(
                    f"Mask folder not found: {mask_dir}"
                )
            if not depth_dir.exists():
                raise FileNotFoundError(
                    f"Depth folder not found: {depth_dir}"
                )

            image_paths = sorted(
                p for p in image_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                }
            )

            for image_path in image_paths:
                mask_path = self._find_matching_file(
                    mask_dir, image_path.stem
                )
                depth_path = depth_dir / f"{image_path.stem}.npy"

                if mask_path is None:
                    raise FileNotFoundError(
                        f"No matching mask for {image_path}"
                    )
                if not depth_path.exists():
                    raise FileNotFoundError(
                        f"No matching depth map: {depth_path}"
                    )

                self.samples.append(
                    (image_path, mask_path, depth_path)
                )

        self.rgb_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(
            f"RGBDepthDataset: {len(self.samples)} samples "
            f"for split='{split}'"
        )

    @staticmethod
    def _find_matching_file(mask_dir, stem):
        # Most datasets use the same filename stem for image and mask.
        for extension in (
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
        ):
            candidate = mask_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, depth_path = self.samples[index]

        # RGB image
        image = Image.open(image_path).convert("RGB")
        rgb = self.rgb_transform(image)

        # Cached relative-depth prediction
        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise ValueError(f"Invalid depth map: {depth_path}")

        # Resize depth using bilinear interpolation.
        depth = cv2.resize(
            depth,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalize each map independently to [0, 1].
        # Constant maps become zeros to avoid division by zero.
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)

        depth_tensor = torch.from_numpy(
            np.ascontiguousarray(depth[None, :, :])
        ).float()

        # Binary ground-truth mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.NEAREST,
        )
        mask = np.asarray(mask, dtype=np.float32) / 255.0
        mask = (mask >= 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask[None, :, :]).float()

        # 3 RGB channels + 1 depth channel = 4 channels
        input_tensor = torch.cat([rgb, depth_tensor], dim=0)

        return input_tensor, mask_tensor
