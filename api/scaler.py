import torch
import torchvision.transforms.functional as TF
from PIL import Image

from models.py.SuperResolution import SupResNet
from models.params.paths import ConstructPath

optimumFeatures = {
    4: 64,
    8: 64,
    16: 64,
}
optimumBlockCount = {
    4: 12,
    8: 12,
    16: 12,
}
optimumLr = {
    4: 1e-4,
    8: 1e-4,
    16: 1e-4,
}


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


def loadModel(scale: int) -> SupResNet:
    """
    Loads a model given a pre-determined optimum configuration for the given scale factor
    This is the simplest and easiest way to load a model for this upscaler.
    These optimum configurations were pre-determined by a hyperparameter search on the Flickr2K dataset.

    :param scale: The upscaling factor
    :type scale: int
    :return: The trained SupResNet model with the optimum configuration
    :rtype: SupResNet
    """
    return loadModel(
        scale, optimumFeatures[scale], optimumBlockCount[scale], optimumLr[scale]
    )


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


def upscaleImage(image: Image.Image, scale: int) -> Image.Image:
    """
    The simplest way to upscale an image.
    Automatically loads the optimum model for a given scale factor, then upscales the image.


    :param image: The image to upscale
    :type image: Image.Image
    :param scale: The scale factor
    :type scale: int
    :return: The final upscaled image
    :rtype: Image
    """
    model = loadModel(scale)
    return upscaleImage(image, scale, model)


def upscaleImage(imagePath: str, scale: int) -> Image.Image:
    """
    The simplest way to upscale an image given a path.
    Automatically loads the optimum model for a given scale factor, then upscales the image located at the specified path.

    :param imagePath: The path to the image to upscale
    :type imagePath: str
    :param scale: The scale factor
    :type scale: int
    :return: The final upscaled image
    :rtype: Image
    """
    model = loadModel(scale)
    return upscaleImage(imagePath, scale, model)
