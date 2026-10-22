from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms


class CannyDataset(Dataset):

    def __init__(
        self,
        dataset_dir,
        split,
        image_size=352
    ):

        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size

        self.image_dir = (
            self.dataset_dir
            / split
            / "images"
        )

        self.mask_dir = (
            self.dataset_dir
            / split
            / "masks"
        )

        self.image_paths = sorted(
            [
                f
                for f in self.image_dir.iterdir()
                if f.is_file()
                and f.suffix.lower()
                in [".jpg", ".jpeg", ".png"]
            ]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        # ------------------------------------------
        # RGB IMAGE TRANSFORM
        # ------------------------------------------

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],

                    std=[
                        0.229,
                        0.224,
                        0.225
                    ]
                )
            ]
        )

        # ------------------------------------------
        # MASK TRANSFORM
        # ------------------------------------------

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=(
                        transforms.InterpolationMode.NEAREST
                    )
                ),

                transforms.ToTensor()
            ]
        )

    def __len__(self):

        return len(
            self.image_paths
        )

    def __getitem__(self, index):

        image_path = (
            self.image_paths[index]
        )

        mask_path = (
            self.mask_dir
            / f"{image_path.stem}.png"
        )

        # ------------------------------------------
        # LOAD RGB IMAGE
        # ------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ------------------------------------------
        # CREATE CANNY EDGE
        # ------------------------------------------

        # Resize first so the Canny image has the
        # same spatial dimensions as the RGB image.

        image_for_canny = image.resize(
            (self.image_size, self.image_size)
        )

        image_np = np.array(
            image_for_canny
        )

        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

        edges = cv2.Canny(
            gray,
            threshold1=100,
            threshold2=200
        )

        # Convert Canny edge map from:
        # 0 / 255
        #
        # to:
        # 0.0 / 1.0

        edges = (
            edges.astype(np.float32)
            / 255.0
        )

        edge_tensor = torch.from_numpy(
            edges
        ).unsqueeze(0)

        # ------------------------------------------
        # RGB TRANSFORM
        # ------------------------------------------

        image_tensor = self.image_transform(
            image
        )

        # ------------------------------------------
        # COMBINE RGB + CANNY
        # ------------------------------------------

        image_tensor = torch.cat(
            [
                image_tensor,
                edge_tensor
            ],
            dim=0
        )

        # ------------------------------------------
        # LOAD MASK
        # ------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = self.mask_transform(
            mask
        )

        # Ensure strictly binary values

        mask = (
            mask > 0.5
        ).float()

        return (
            image_tensor,
            mask
        )