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


def upscaleImage():
    pass
