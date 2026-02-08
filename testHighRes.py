import torch

from PIL.Image import Resampling

from api import scaler
from api.utility import simulateLowRes, loadImage
from api.plots import imageDiagram


def run_test(
    inputPath: str,
    scale: int = 4,
):
    print("Upscaling starting...")

    highResImage = loadImage(inputPath)

    lowResImage, lowResUpscaledImage = simulateLowRes(
        highResImage, scale, Resampling.NEAREST
    )

    modelImage = scaler.upscaleImage(lowResImage, scale)

    imageDiagram(lowResImage, modelImage, "Low Res", "Model")

    # imageDiagram(
    #     highResImage, lowResUpscaledImage, "High Res", "Low Res", modelImage, "Model"
    # )

    print("Upscaling complete!")


if __name__ == "__main__":
    run_test(
        inputPath="images/knight.png",
        scale=4,
    )
