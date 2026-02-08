import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional


def imageDiagram(
    image1: Image.Image,
    image2: Image.Image,
    title1: str,
    title2: str,
    image3: Optional[Image.Image] = None,
    title3: Optional[str] = None,
):
    """
    Docstring for plotImage

    :param image1: PIL Image 1
    :type image1: Image.Image
    :param image2: PIL Image 1
    :type image2: Image.Image
    :param title1: Title for Image 1
    :type title1: str
    :param title2: Title for Image 2
    :type title2: str
    :param image3: PIL Image 3 (Optional)
    :type image3: Optional[Image.Image]
    :param title3: Title for Image 3 (Optional)
    :type title3: Optional[str]
    """
    if (image3 is None and title3 is not None) or (
        image3 is not None and title3 is None
    ):
        raise ValueError

    cols = 3 if image3 is not None else 2
    imageSize = 4

    plt.figure(figsize=(cols * imageSize, imageSize))

    # Image 1
    plt.subplot(1, cols, 1)
    plt.title(title1)
    plt.imshow(image1)
    plt.axis("off")

    # Image 2
    plt.subplot(1, cols, 2)
    plt.title(title2)
    plt.imshow(image2)
    plt.axis("off")

    # Optional image 3
    if image3 is not None:
        plt.subplot(1, cols, 3)
        plt.title(title3)
        plt.imshow(image3)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
