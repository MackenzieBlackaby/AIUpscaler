import torch
import torchvision.transforms.functional as TF
from PIL import Image

from models.py.SuperResolution import SupResNet
from models.params.paths import ConstructPath


def loadModel(scale: int, features: int, blockCount: int, lr: float) -> SupResNet:
    """
    Loads a model and the checkpoint given the data about the model.

    :param scale: The upscaling factor
    :type scale: int
    :param features: The feature count of the model (micro complexity)
    :type features: int
    :param blockCount: The block count of the model (macro complexity)
    :type blockCount: int
    :param lr: The learning rate of the model
    :type lr: float
    :return: The trained SupResNet model with specified configuration
    :rtype: SupResNet
    """
    model = SupResNet(scale=scale, blockCount=blockCount, features=features)
    ckpt = torch.load(ConstructPath(scale, features, blockCount, lr))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def upscaleImage(image: Image.Image, scale: int, model: SupResNet) -> Image.Image:
    """
    Upscales an image to the desired scale factor using Super Resolution upscaling.

    :param image: The image to upscale.
    :type image: Image.Image
    :param scale: The scale factor
    :type scale: int
    :param model: The model to be used for upscaling
    :type model: SupResNet
    :return: The final upscaled image
    :rtype: Image.Image
    """
    raise NotImplemented


def upscaleImage(imagePath: str, scale: int, model: SupResNet) -> Image.Image:
    """
    Upscales an image located at a specified path to the desired scale factor using Super Resolution upscaling.

    :param imagePath: The path to the image to upscale
    :type imagePath: str
    :param scale: The scale factor
    :type scale: int
    :param model: The model to be used for upscaling
    :type model: SupResNet
    :return: The final upscaled image
    :rtype: Image.Image
    """
    image = Image.open(imagePath).convert("RGB")
    return upscaleImage(image, scale, model)
