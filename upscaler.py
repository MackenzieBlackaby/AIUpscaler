import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

from models.py.SuperResolution import SupResNet


def load_model(ckpt_path: str, scale: int, device: torch.device):
    model = SupResNet(scale=scale, blockCount=12, features=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


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
    ckpt_path: str,
    input_path: str,
    scale: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load model
    model = load_model(ckpt_path, scale=scale, device=device)

    # 2) Load HR image
    hr_pil = Image.open(input_path).convert("RGB")

    # 3) Simulate LR input
    lr_pil = simulate_lr_pil(hr_pil, scale=scale)

    # For fair visual comparison, upsample the LR back to HR size (bicubic)
    lr_up_pil = lr_pil.resize(hr_pil.size, resample=Image.BICUBIC)

    # 4) PIL -> tensor in [0,1], shape [1,C,H,W]
    lr_tensor = TF.to_tensor(lr_pil).unsqueeze(0).to(device)

    # 5) Forward pass -> SR tensor [1,C,H,W] in roughly [0,1]
    sr_tensor = model(lr_tensor).clamp(0.0, 1.0).cpu().squeeze(0)

    # 6) Convert SR tensor to PIL for display
    sr_pil = TF.to_pil_image(sr_tensor)

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
        ckpt_path="sr_last.pt",
        input_path="testimages/knight.png",  # change this
        scale=4,  # must match training
    )
