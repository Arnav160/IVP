
import segmentation_models_pytorch as smp


def create_model():
    """
    U-Net with a pretrained ResNet-34 encoder.
    Input: RGB (3 channels) + relative depth (1 channel).
    Output: one segmentation logit per pixel.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
        activation=None,
    )

    return model
