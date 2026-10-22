from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import io
import base64

# Add project root to path for importing model utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# Import Depth Anything V2
# Go up two levels: website -> project -> IVP_projevt (where Depth-Anything-V2 is)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
depth_anything_path = os.path.join(os.path.dirname(project_root), 'Depth-Anything-V2')
sys.path.insert(0, depth_anything_path)
from depth_anything_v2.dpt import DepthAnythingV2

# Import U-Net model
from models.unet_resnet_depth.model import create_model as create_depth_model

app = Flask(__name__)

# Global model instances (loaded on startup)
seg_model = None
depth_model = None
device = None

# Image preprocessing for segmentation model
SEG_IMAGE_SIZE = 352
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

rgb_transform = transforms.Compose([
    transforms.Resize(
        (SEG_IMAGE_SIZE, SEG_IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
])


def load_models():
    """Load both Depth Anything V2 and segmentation models."""
    global seg_model, depth_model, device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Depth Anything V2 (ViT-Small for speed)
    print("Loading Depth Anything V2...")
    depth_model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    depth_model = DepthAnythingV2(**depth_model_config)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    depth_checkpoint = os.path.join(os.path.dirname(project_root), 'Depth-Anything-V2', 'checkpoints', 'depth_anything_v2_vits.pth')
    depth_model.load_state_dict(torch.load(depth_checkpoint, map_location='cpu'))
    depth_model = depth_model.to(device).eval()
    print("Depth Anything V2 loaded!")

    # Load segmentation model (RGB + Depth)
    print("Loading segmentation model...")
    seg_model = create_depth_model()
    seg_checkpoint = os.path.join(project_root, 'checkpoints', 'unet_resnet34_depth_best.pth')
    checkpoint = torch.load(seg_checkpoint, map_location='cpu')
    # Checkpoint contains model_state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    seg_model.load_state_dict(state_dict)
    seg_model = seg_model.to(device).eval()
    print("Segmentation model loaded!")


def generate_depth_map(image):
    """Generate depth map using Depth Anything V2."""
    # image: numpy array (H, W, 3) in BGR format from OpenCV
    # Convert to RGB for Depth Anything V2
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Infer depth (returns H x W numpy array)
    depth = depth_model.infer_image(image_rgb, input_size=518)

    # Normalize to [0, 1] per image (same as training)
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)

    return depth.astype(np.float32)


def preprocess_for_segmentation(rgb_image, depth_map):
    """
    Prepare RGB + depth input for segmentation model.

    Args:
        rgb_image: PIL Image (RGB)
        depth_map: numpy array (H, W) normalized to [0, 1]

    Returns:
        input_tensor: torch.Tensor (1, 4, 352, 352)
    """
    # Transform RGB
    rgb_tensor = rgb_transform(rgb_image)  # (3, 352, 352)

    # Resize depth to 352x352 using bilinear interpolation
    depth_resized = cv2.resize(
        depth_map,
        (SEG_IMAGE_SIZE, SEG_IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # Depth is already normalized to [0, 1] from generate_depth_map
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depth_resized[None, :, :])).float()  # (1, 352, 352)

    # Concatenate: 3 RGB + 1 depth = 4 channels
    input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0).unsqueeze(0)  # (1, 4, 352, 352)

    return input_tensor


def tensor_to_base64(tensor, is_mask=False):
    """Convert tensor to base64 encoded PNG for frontend display."""
    if is_mask:
        # Mask: (1, 1, H, W) -> (H, W) uint8
        img_array = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array, mode='L')
    else:
        # RGB: (3, H, W) normalized -> (H, W, 3) uint8
        img_array = tensor.squeeze().cpu().numpy()
        # Denormalize
        for i in range(3):
            img_array[i] = img_array[i] * RGB_STD[i] + RGB_MEAN[i]
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        img_array = np.transpose(img_array, (1, 2, 0))  # (H, W, 3)
        pil_img = Image.fromarray(img_array, mode='RGB')

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_overlay_base64(rgb_tensor, mask_tensor, alpha=0.5):
    """Create overlay of mask on RGB image."""
    # Denormalize RGB
    rgb_np = rgb_tensor.squeeze().cpu().numpy()
    for i in range(3):
        rgb_np[i] = rgb_np[i] * RGB_STD[i] + RGB_MEAN[i]
    rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
    rgb_np = np.transpose(rgb_np, (1, 2, 0))  # (H, W, 3)

    # Mask
    mask_np = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_color = np.zeros_like(rgb_np)
    mask_color[:, :, 0] = mask_np  # Red channel for mask

    # Blend
    overlay = cv2.addWeighted(rgb_np, 1.0, mask_color, alpha, 0)

    pil_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.route('/')
def index():
    """Home / Dashboard page"""
    return render_template('index.html')


@app.route('/predict')
def predict():
    """Prediction page"""
    return render_template('predict.html')


@app.route('/results')
def results():
    """Results / Model Comparison page"""
    return render_template('results.html')


@app.route('/about')
def about():
    """About / Methodology page"""
    return render_template('about.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for segmentation prediction."""
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No image selected'})

    try:
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return jsonify({'status': 'error', 'message': 'Invalid image file'})

        # Convert to PIL for RGB transform
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Generate depth map
        depth_map = generate_depth_map(image_bgr)

        # Preprocess for segmentation
        input_tensor = preprocess_for_segmentation(pil_image, depth_map)
        input_tensor = input_tensor.to(device)

        # Run inference
        with torch.no_grad():
            logits = seg_model(input_tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()

        # Prepare response images
        rgb_base64 = tensor_to_base64(input_tensor[:, :3, :, :])
        mask_base64 = tensor_to_base64(pred_mask, is_mask=True)
        overlay_base64 = create_overlay_base64(input_tensor[:, :3, :, :], pred_mask)

        return jsonify({
            'status': 'success',
            'rgb': f'data:image/png;base64,{rgb_base64}',
            'mask': f'data:image/png;base64,{mask_base64}',
            'overlay': f'data:image/png;base64,{overlay_base64}',
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    # Load models on startup
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)