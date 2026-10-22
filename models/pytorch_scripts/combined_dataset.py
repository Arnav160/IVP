
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


class CombinedDataset(Dataset):
    """
    Dataset for RGB + Canny + Depth camouflaged-object segmentation.

    Input:
        RGB:    3 channels
        Canny:  1 channel
        Depth:  1 channel
        Total:  5 channels, shape [5, H, W]

    Target:
        Binary mask, shape [1, H, W]
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(
        self,
        project_dir,
        datasets=("COD10K_CAM", "CAMO", "NC4K"),
        split="train",
        image_size=352,
    ):
        self.project_dir = Path(project_dir)
        self.datasets = datasets
        self.split = split
        self.image_size = image_size
        self.samples = []

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        for dataset_name in datasets:
            split_dir = (
                self.project_dir / "datasets" / dataset_name / split
            )
            image_dir = split_dir / "images"
            mask_dir = split_dir / "masks"
            depth_dir = (
                self.project_dir / "depth_maps" / dataset_name / split
            )

            for folder in (image_dir, mask_dir, depth_dir):
                if not folder.is_dir():
                    raise FileNotFoundError(
                        f"Required directory not found: {folder}"
                    )

            # Index masks by filename stem, independent of extension.
            mask_index = {}
            for mask_path in mask_dir.iterdir():
                if (
                    mask_path.is_file()
                    and mask_path.suffix.lower()
                    in self.IMAGE_EXTENSIONS
                ):
                    mask_index.setdefault(
                        mask_path.stem.lower(), []
                    ).append(mask_path)

            image_paths = sorted(
                path for path in image_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in self.IMAGE_EXTENSIONS
            )

            for image_path in image_paths:
                matching_masks = mask_index.get(
                    image_path.stem.lower(), []
                )

                if not matching_masks:
                    raise FileNotFoundError(
                        f"No mask matches image '{image_path.name}'. "
                        f"Looked by filename stem in: {mask_dir}"
                    )

                if len(matching_masks) > 1:
                    # Prefer the common mask extensions in this order.
                    matching_masks.sort(
                        key=lambda p: (
                            self.MASK_EXTENSIONS.index(p.suffix.lower())
                            if p.suffix.lower() in self.MASK_EXTENSIONS
                            else 999
                        )
                    )

                mask_path = matching_masks[0]
                depth_path = depth_dir / f"{image_path.stem}.npy"

                if not depth_path.is_file():
                    raise FileNotFoundError(
                        f"Depth map missing for '{image_path.name}': "
                        f"{depth_path}"
                    )

                self.samples.append(
                    (
                        dataset_name,
                        image_path,
                        mask_path,
                        depth_path,
                    )
                )

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split '{split}'."
            )

        print(
            f"CombinedDataset: {len(self.samples)} samples "
            f"for split='{split}'"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        dataset_name, image_path, mask_path, depth_path = (
            self.samples[index]
        )

        # ----------------------------------
        # 1. RGB image: 3 channels
        # ----------------------------------
        image = Image.open(image_path).convert("RGB")
        rgb_tensor = self.image_transform(image)

        # ----------------------------------
        # 2. Canny edge map: 1 channel
        # ----------------------------------
        image_cv = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_cv is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            threshold1=100,
            threshold2=200,
        )

        edges = cv2.resize(
            edges,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        edges_tensor = torch.from_numpy(
            edges.astype(np.float32) / 255.0
        ).unsqueeze(0)

        # ----------------------------------
        # 3. Cached depth map: 1 channel
        # ----------------------------------
        depth = np.load(depth_path).astype(np.float32)

        if depth.ndim != 2:
            raise ValueError(
                f"Expected a 2D depth map, got {depth.shape}: "
                f"{depth_path}"
            )

        if not np.isfinite(depth).all():
            raise ValueError(
                f"Depth map contains NaN or infinite values: "
                f"{depth_path}"
            )

        depth = cv2.resize(
            depth,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalize each depth map to [0, 1].
        depth_min = float(depth.min())
        depth_max = float(depth.max())

        if depth_max > depth_min:
            depth = (
                (depth - depth_min) / (depth_max - depth_min)
            )
        else:
            depth = np.zeros_like(depth, dtype=np.float32)

        depth_tensor = torch.from_numpy(
            np.ascontiguousarray(depth, dtype=np.float32)
        ).unsqueeze(0)

        # ----------------------------------
        # 4. Binary ground-truth mask
        # ----------------------------------
        mask = Image.open(mask_path).convert("L")
        mask = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST,
        )(mask)

        mask_array = np.array(mask, dtype=np.uint8)
        mask_array = (mask_array > 127).astype(np.float32)

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0)

        # ----------------------------------
        # 5. Combine RGB + Canny + Depth
        # ----------------------------------
        input_tensor = torch.cat(
            [
                rgb_tensor,
                edges_tensor,
                depth_tensor,
            ],
            dim=0,
        )

        if input_tensor.shape != (
            5, self.image_size, self.image_size
        ):
            raise RuntimeError(
                f"Unexpected input shape: {input_tensor.shape}"
            )

        return input_tensor, mask_tensor
