import sys
from pathlib import Path

import torch

# Add project folders to Python path
PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_DIR / "pytorch")
)

from canny_dataset import CannyDataset
from canny_model import create_canny_model


# ------------------------------------------
# CONFIG
# ------------------------------------------

DATASET_DIR = (
    PROJECT_DIR
    / "datasets"
    / "COD10K_CAM"
)

IMAGE_SIZE = 352


# ------------------------------------------
# DEVICE
# ------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ------------------------------------------
# DATASET
# ------------------------------------------

dataset = CannyDataset(
    dataset_dir=DATASET_DIR,
    split="train",
    image_size=IMAGE_SIZE
)

print(
    "Dataset size:",
    len(dataset)
)


# ------------------------------------------
# GET ONE SAMPLE
# ------------------------------------------

image, mask = dataset[0]

print(
    "Image shape:",
    image.shape
)

print(
    "Mask shape:",
    mask.shape
)

print(
    "Image dtype:",
    image.dtype
)

print(
    "Mask dtype:",
    mask.dtype
)

print(
    "Image min:",
    image.min().item()
)

print(
    "Image max:",
    image.max().item()
)

print(
    "Mask unique values:",
    torch.unique(mask)
)


# ------------------------------------------
# MODEL
# ------------------------------------------

model = create_canny_model()

model = model.to(device)

print(
    "Model created successfully."
)


# ------------------------------------------
# TEST FORWARD PASS
# ------------------------------------------

image_batch = image.unsqueeze(0)

image_batch = image_batch.to(device)

with torch.no_grad():

    output = model(
        image_batch
    )

print(
    "Input shape:",
    image_batch.shape
)

print(
    "Output shape:",
    output.shape
)


# ------------------------------------------
# FINAL CHECK
# ------------------------------------------

expected_input = (
    1,
    4,
    IMAGE_SIZE,
    IMAGE_SIZE
)

expected_output = (
    1,
    1,
    IMAGE_SIZE,
    IMAGE_SIZE
)

assert image_batch.shape == expected_input

assert output.shape == expected_output

print()
print("====================================")
print("CANNY MODEL TEST PASSED")
print("====================================")