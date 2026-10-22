
import segmentation_models_pytorch as smp


def create_model():
    """
    U-Net with a ResNet-34 encoder for camouflaged object segmentation.

    Input channels:
        0-2: RGB
        3:   Canny edge map
        4:   Depth map

    Output:
        One segmentation logit per pixel.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=5,
        classes=1,
        activation=None,
    )

    return model
