import torch
import torchvision.transforms.functional as TF

from PIL import Image
from typing import Optional, Union

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
trainedScales = [2, 3, 4]


def getDevice() -> torch.device:
    """
    Gets the current device being used.

    :return: Device being used
    :rtype: device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadModel(
    scale: int,
    features: Optional[int] = None,
    blockCount: Optional[int] = None,
    lr: Optional[float] = None,
) -> SupResNet:
    """
    Loads a model and the checkpoint given the data about the model.

    If features, blockCount, and lr are not provided, the pre-determined optimum configuration
    for the given scale factor is used (from the hyperparameter search on Flickr2K).

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
    # TODO: Add checking that scale is in a list of trained configurations. If not, repeatedly scale the image to achieve desired scaling

    if features is None or blockCount is None or lr is None:
        features = optimumFeatures[scale]
        blockCount = optimumBlockCount[scale]
        lr = optimumLr[scale]

    model = SupResNet(scale=scale, blockCount=blockCount, features=features)
    ckpt = torch.load(
        ConstructPath(scale, features, blockCount, lr), map_location=getDevice()
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def upscaleImage(
    image: Union[Image.Image, str],
    scale: int,
    model: Optional[SupResNet] = None,
) -> Image.Image:
    """
    Upscales an image to the desired scale factor using Super Resolution upscaling.

    If `image` is a string, it is treated as a path to an image.
    If `model` is not provided, the optimum model for the given scale factor is loaded automatically.

    :param image: The image to upscale, or a path to the image.
    :type image: Image.Image | str
    :param scale: The scale factor
    :type scale: int
    :param model: The model to be used for upscaling
    :type model: SupResNet
    :return: The final upscaled image
    :rtype: Image.Image
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if model is None:
        model = loadModel(scale)

    imageTensor = TF.to_tensor(image).unsqueeze(0).to(getDevice())
    upscaledTensor = model(imageTensor).clamp(0.0, 1.0).cpu().squeeze(0)
    upscaledImage = TF.to_pil_image(upscaledTensor)
    return upscaledImage
