
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from depth_dataset import RGBDepthDataset

PROJECT_DIR = Path(
    r"C:\Users\Arnav Agrawal\Desktop\college\IVP_projevt\project"
)

dataset = RGBDepthDataset(
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

images, masks = next(iter(loader))

print("Input batch shape:", images.shape)
print("Mask batch shape:", masks.shape)
print("Input dtype:", images.dtype)
print("Mask values:", torch.unique(masks))
print("Depth channel range:",
      images[:, 3].min().item(),
      images[:, 3].max().item())

assert images.shape == (2, 4, 352, 352)
assert masks.shape == (2, 1, 352, 352)
assert torch.isfinite(images).all()
assert torch.isfinite(masks).all()

print("Dataset test passed!")
