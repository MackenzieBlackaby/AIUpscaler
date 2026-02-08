import torch
from PIL import Image
from PIL.Image import Resampling
from api import scaler
from api.utility import simulateLowRes, loadImage
from api.plots import imageDiagram


@torch.no_grad()
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
        inputPath="ignore/testimages/knight.png",
        scale=4,
    )
