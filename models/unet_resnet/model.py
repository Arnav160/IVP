import segmentation_models_pytorch as smp


def create_model():

    model = smp.Unet(

        # ResNet encoder
        encoder_name="resnet34",

        # Use ImageNet pretrained encoder weights
        encoder_weights="imagenet",

        # RGB input
        in_channels=3,

        # Binary segmentation output
        classes=1,

        # Raw logits output
        activation=None
    )

    return model