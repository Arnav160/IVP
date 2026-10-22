
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from combined_dataset import CombinedDataset


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    dataset = CombinedDataset(
        project_dir=PROJECT_DIR,
        split="train",
        image_size=352,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    inputs, masks = next(iter(loader))

    print("Input batch shape:", inputs.shape)
    print("Mask batch shape:", masks.shape)
    print("Input dtype:", inputs.dtype)
    print("Mask values:", torch.unique(masks))

    print(
        "RGB range:",
        inputs[:, :3].min().item(),
        inputs[:, :3].max().item(),
    )
    print(
        "Canny range:",
        inputs[:, 3].min().item(),
        inputs[:, 3].max().item(),
    )
    print(
        "Depth range:",
        inputs[:, 4].min().item(),
        inputs[:, 4].max().item(),
    )

    assert inputs.shape == (2, 5, 352, 352)
    assert masks.shape == (2, 1, 352, 352)
    assert torch.isfinite(inputs).all()
    assert torch.isfinite(masks).all()
    assert set(torch.unique(masks).tolist()).issubset({0.0, 1.0})
    assert inputs[:, 3].min() >= 0 and inputs[:, 3].max() <= 1
    assert inputs[:, 4].min() >= 0 and inputs[:, 4].max() <= 1

    print("\nCombined dataset test passed!")


if __name__ == "__main__":
    main()
