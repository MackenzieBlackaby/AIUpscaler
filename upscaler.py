import torch
from PIL import Image
import matplotlib.pyplot as plt
from api import scaler


def simulate_lr_pil(hr_img: Image.Image, scale: int) -> Image.Image:
    """
    Simulate a low-res image by downscaling the HR image by `scale` with bicubic.
    Returns a PIL Image at low resolution.
    """
    w, h = hr_img.size
    lr_w, lr_h = max(1, w // scale), max(1, h // scale)
    lr_img = hr_img.resize((lr_w, lr_h), resample=Image.BICUBIC)
    return lr_img


@torch.no_grad()
def run_test(
    input_path: str,
    scale: int = 4,
):
    print("Upscaling starting...")

    # 2) Load HR image
    hr_pil = Image.open(input_path).convert("RGB")

    # 3) Simulate LR input
    lr_pil = simulate_lr_pil(hr_pil, scale=scale)

    # For fair visual comparison, upsample the LR back to HR size (bicubic)
    lr_up_pil = lr_pil.resize(hr_pil.size, resample=Image.BICUBIC)

    sr_pil = scaler.upscaleImage(lr_pil, scale)

    # 7) Plot comparison
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original (HR)")
    plt.imshow(hr_pil)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Simulated LR (↓{scale} then ↑ bicubic)")
    plt.imshow(lr_up_pil)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Super-Res (Model Output)")
    plt.imshow(sr_pil)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test(
        input_path="ignore/testimages/knight.png",
        scale=4,
    )
