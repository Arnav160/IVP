import segmentation_models_pytorch as smp


def create_canny_model():

    model = smp.Unet(

        # ResNet-34 encoder
        encoder_name="resnet34",

        # ImageNet pretrained encoder
        encoder_weights="imagenet",

        # RGB (3) + Canny (1) = 4 channels
        in_channels=4,

        # Binary segmentation
        classes=1,

        # Raw logits
        activation=None
    )

    return model