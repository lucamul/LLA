import segmentation_models_pytorch as smp

def deeplab_model(encoder = 'resnet34'):
    """ Fetches pre trained model from deep lab

    Args:
        encoder (str, optional): encoder used. Defaults to 'resnet34'.

    Returns:
        torch.nn.Module : pre trained model
    """
    return smp.DeepLabV3(encoder_name=encoder, encoder_depth=5, encoder_weights="imagenet",in_channels=3,classes=1)