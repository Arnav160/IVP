import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import time
import os
from pathlib import Path

# Import our web modules
from web.inference import load_rgb_model, load_canny_model, predict_rgb, predict_canny
from web.utils import preprocess_image, create_overlay, get_model_info

# Page configuration
st.set_page_config(
    page_title="Camouflaged Object Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .model-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .upload-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">CAMOUFLAGED OBJECT SEGMENTATION</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">U-Net with ResNet-34</div>', unsafe_allow_html=True)

# Sidebar for model selection and info
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        ["RGB U-Net (ResNet-34)", "RGB + Canny U-Net (ResNet-34)"],
        index=0
    )

    # Model info
    st.subheader("📊 Model Information")
    model_info = get_model_info(model_choice)
    st.markdown(f'<div class="model-info">{model_info}</div>', unsafe_allow_html=True)

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        threshold = st.slider(
            "Prediction Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Threshold for converting probabilities to binary mask"
        )

    # About section
    st.markdown("---")
    st.markdown("""
    ### About
    This application demonstrates camouflaged object segmentation using:
    - **RGB U-Net**: Standard U-Net with ResNet-34 encoder
    - **RGB + Canny U-Net**: U-Net with RGB and Canny edge channels

    Models are trained on COD10K-CAM, CAMO, and NC4K datasets.
    """)

# Main content area
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to segment camouflaged objects"
    )

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

with col2:
    st.subheader("🎯 Predicted Mask")
    if uploaded_file is not None:
        # Placeholder for prediction
        with st.spinner("Processing image..."):
            start_time = time.time()

            # Preprocess image
            image_tensor, original_size = preprocess_image(image)

            # Run inference based on model selection
            if model_choice == "RGB U-Net (ResNet-34)":
                model = load_rgb_model()
                prediction = predict_rgb(model, image_tensor, threshold)
            else:  # RGB + Canny U-Net
                model = load_canny_model()
                prediction = predict_canny(model, image_tensor, threshold)

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Display prediction
            st.image(prediction, caption="Segmentation Mask", use_column_width=True, clamp=True)

with col3:
    st.subheader("👁️ Overlay")
    if uploaded_file is not None:
        # Create and display overlay
        overlay = create_overlay(image, prediction)
        st.image(overlay, caption="Prediction Overlay", use_column_width=True)

        # Display inference time
        st.markdown(f'<div class="upload-info">⏱️ Inference Time: {inference_time:.1f} ms</div>', unsafe_allow_html=True)

# Model performance info (only show when image is uploaded)
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("📈 Model Performance (on test datasets)")

    # Performance metrics from the project documentation
    if model_choice == "RGB U-Net (ResNet-34)":
        perf_data = {
            "Dataset": ["COD10K-CAM", "CAMO", "NC4K"],
            "Dice Score": [0.638466, 0.597130, 0.755478],
            "IoU Score": [0.525149, 0.484481, 0.650747]
        }
    else:  # Canny model
        perf_data = {
            "Dataset": ["COD10K-CAM", "CAMO", "NC4K"],
            "Dice Score": [0.625301, 0.659307, 0.762712],
            "IoU Score": [0.504959, 0.541913, 0.653384]
        }

    st.table(perf_data)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Camouflaged Object Segmentation Project | Built with Streamlit & PyTorch"
    "</div>",
    unsafe_allow_html=True
)