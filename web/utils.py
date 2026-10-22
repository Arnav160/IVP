import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def preprocess_image(pil_image):
    """
    Preprocess a PIL image for model inference.
    Applies the same preprocessing as used in training:
    - Resize to 352x352
    - Convert to tensor
    - Apply ImageNet normalization

    Args:
        pil_image: PIL Image object (RGB)

    Returns:
        image_tensor: Preprocessed tensor of shape [3, 352, 352]
        original_size: Tuple of (width, height) of original image
    """
    original_size = pil_image.size  # (width, height)

    # Define the same transform as in training
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transform
    image_tensor = transform(pil_image)

    return image_tensor, original_size

def create_overlay(original_image, prediction_mask, alpha=0.4, color=[255, 0, 0]):
    """
    Create an overlay of the prediction mask on the original image.

    Args:
        original_image: PIL Image object (RGB)
        prediction_mask: numpy array of shape [H, W] with values 0 or 255
        alpha: Transparency factor for the overlay (0.0 to 1.0)
        color: RGB color for the mask overlay [R, G, B]

    Returns:
        overlay: PIL Image object with the overlay applied
    """
    # Convert PIL image to numpy array
    original_np = np.array(original_image)

    # Resize prediction mask to match original image size
    mask_resized = cv2.resize(
        prediction_mask,
        original_np.shape[1::-1],  # (width, height)
        interpolation=cv2.INTER_NEAREST
    )

    # Create colored mask
    colored_mask = np.zeros_like(original_np)
    colored_mask[mask_resized > 0] = color

    # Create overlay
    overlay = cv2.addWeighted(original_np, 1 - alpha, colored_mask, alpha, 0)

    # Convert back to PIL Image
    return Image.fromarray(overlay)

def get_model_info(model_choice):
    """
    Get information about the selected model for display in the sidebar.

    Args:
        model_choice: String indicating the selected model

    Returns:
        info_string: HTML-formatted string with model information
    """
    if model_choice == "RGB U-Net (ResNet-34)":
        return """
        <strong>Architecture:</strong> U-Net with ResNet-34 encoder<br>
        <strong>Input Channels:</strong> 3 (RGB)<br>
        <strong>Input Size:</strong> 352 × 352<br>
        <strong>Output:</strong> Binary segmentation mask<br>
        <strong>Parameters:</strong> ~24.4 million<br>
        <strong>Training:</strong> Combined COD10K-CAM + CAMO + NC4K<br>
        <strong>Best Checkpoint:</strong> unet_resnet34_all_datasets_best.pth
        """
    else:  # RGB + Canny U-Net
        return """
        <strong>Architecture:</strong> U-Net with ResNet-34 encoder<br>
        <strong>Input Channels:</strong> 4 (RGB + Canny)<br>
        <strong>Input Size:</strong> 352 × 352<br>
        <strong>Output:</strong> Binary segmentation mask<br>
        <strong>Parameters:</strong> ~24.4 million<br>
        <strong>Canny Edge:</strong> cv2.Canny(gray, 100, 200)<br>
        <strong>Training:</strong> Extended training on combined datasets<br>
        <strong>Best Checkpoint:</strong> unet_resnet34_canny_extended_best.pth
        """