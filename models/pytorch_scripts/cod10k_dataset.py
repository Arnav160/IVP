from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms


class COD10KDataset(Dataset):

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
        # IMAGE TRANSFORM
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
        # LOAD IMAGE
        # ------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ------------------------------------------
        # LOAD MASK
        # ------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        # ------------------------------------------
        # APPLY TRANSFORMS
        # ------------------------------------------

        image = self.image_transform(
            image
        )

        mask = self.mask_transform(
            mask
        )

        # Ensure strictly binary values
        mask = (
            mask > 0.5
        ).float()

        return (
            image,
            mask
        )