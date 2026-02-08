import torch
from PIL import Image
from PIL.Image import Resampling
import matplotlib.pyplot as plt
from api import scaler
from api.utility import simulateLowRes


@torch.no_grad()
def run_test(
    inputPath: str,
    scale: int = 4,
):
    print("Upscaling starting...")

    # 2) Load HR image
    highResImage = Image.open(inputPath).convert("RGB")

    # 3) Simulate LR input
    lowResImage, lowResUpscaledImage = simulateLowRes(
        highResImage, scale, Resampling.BICUBIC
    )

    modelImage = scaler.upscaleImage(lowResImage, scale)

    # 7) Plot comparison
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original (HR)")
    plt.imshow(highResImage)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Simulated LR (↓{scale} then ↑ bicubic)")
    plt.imshow(lowResUpscaledImage)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Super-Res (Model Output)")
    plt.imshow(modelImage)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test(
        inputPath="ignore/testimages/knight.png",
        scale=4,
    )
