from PIL import Image
from PIL.Image import Resampling
from typing import Optional


def simulateLowRes(
    highResImage: Image.Image,
    scale: int,
    upscalingStyle: Optional[Resampling] = Resampling.BICUBIC,
    downscalingStyle: Optional[Resampling] = Resampling.BICUBIC,
) -> tuple[Image.Image, Image.Image]:
    """
    Simulates a low resolution image by downscaling a high resolution image, then upscaling with a specified upscaling style

    :param highResImage: The original high resolution image
    :type highResImage: Image.Image
    :param scale: The factor to downscale the image by
    :type scale: int
    :param upscalingStyle: The PIL Image function to upscale the image with. Default is BICUBIC
    :type upscalingStyle: Resampling
    :param downscalingStyle: (BETTER TO LEAVE DEFAULT) The PIL Image function to downscale the image with. Default is Bicubic
    :type downscalingStyle: Resampling
    :return: A tuple of the low resolution image and the upscaled low resolution image
    :rtype: tuple[Image.Image, Image.Image]
    """
    width, height = highResImage.size
    lowResWidth, lowResHeight = max(1, width // scale), max(1, height // scale)
    lowResImage = highResImage.resize(
        (lowResWidth, lowResHeight), resample=downscalingStyle
    )
    lowResUpscaledImage = lowResImage.resize((width, height), resample=upscalingStyle)
    return (lowResImage, lowResUpscaledImage)
