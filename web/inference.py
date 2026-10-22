import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

# Add project root to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import model creation functions
from models.unet_resnet.model import create_model
from models.unet_resnet.canny_model import create_canny_model

# Global variables for model caching
_rgb_model = None
_canny_model = None
_device = None

def get_device():
    """Get the appropriate device (CUDA if available, else CPU)"""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def load_rgb_model():
    """Load and return the RGB U-Net model"""
    global _rgb_model
    if _rgb_model is None:
        model = create_model()
        model.to(get_device())

        # Load the best checkpoint
        checkpoint_path = PROJECT_ROOT / "checkpoints" / "unet_resnet34_all_datasets_best.pth"
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=get_device()))
            model.eval()
        else:
            raise FileNotFoundError(f"RGB model checkpoint not found at {checkpoint_path}")

        _rgb_model = model
    return _rgb_model

def load_canny_model():
    """Load and return the RGB+Canny U-Net model"""
    global _canny_model
    if _canny_model is None:
        model = create_canny_model()
        model.to(get_device())

        # Load the best extended checkpoint
        checkpoint_path = PROJECT_ROOT / "checkpoints" / "unet_resnet34_canny_extended_best.pth"
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=get_device()))
            model.eval()
        else:
            raise FileNotFoundError(f"Canny model checkpoint not found at {checkpoint_path}")

        _canny_model = model
    return _canny_model

def predict_rgb(model, image_tensor, threshold=0.5):
    """Run inference on RGB model"""
    with torch.no_grad():
        # Move to device
        image_tensor = image_tensor.to(get_device())

        # Forward pass
        logits = model(image_tensor.unsqueeze(0))  # Add batch dimension
        probs = torch.sigmoid(logits)

        # Apply threshold and convert to numpy
        mask = (probs > threshold).float()
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions

        # Convert to uint8 for display (0 or 255)
        mask = (mask * 255).astype(np.uint8)

        return mask

def predict_canny(model, image_tensor, threshold=0.5):
    """Run inference on RGB+Canny model"""
    with torch.no_grad():
        # Move to device
        image_tensor = image_tensor.to(get_device())

        # Forward pass
        logits = model(image_tensor.unsqueeze(0))  # Add batch dimension
        probs = torch.sigmoid(logits)

        # Apply threshold and convert to numpy
        mask = (probs > threshold).float()
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions

        # Convert to uint8 for display (0 or 255)
        mask = (mask * 255).astype(np.uint8)

        return mask