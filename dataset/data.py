import torchvision.transforms.functional as TF

from random import randint
from PIL import Image
from PIL.Image import Resampling
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from kagglehub import dataset_download


# Downloads the data on the fly. Default flickr2k dataset
def DownloadData(name: str = "daehoyang/flickr2k"):
    path = dataset_download("daehoyang/flickr2k")
    return path


if __name__ == "__main__":
    DownloadData()
    pass


class ImagePair:
    def __init__(self, lowRes: Tensor, highRes: Tensor):
        self.lowRes = lowRes
        self.highRes = highRes


class ImageSet(Dataset):
    def __init__(self, rootDir: str, scale: int = 2, hrCrop: int = 192):
        self.suffixes = {".jpg", ".png", ".jpeg", ".webp"}
        self.paths = sorted(
            [p for p in Path(rootDir).rglob("*") if p.suffix.lower() in self.suffixes]
        )
        self.scale = scale
        self.hrCrop = hrCrop

    def __len__(self):
        return len(self.paths)

    # Returns low res and high res tensor pair
    def __getitem__(self, idx):
        # Get img
        img = Image.open(self.paths[idx]).convert("RGB")

        # Size up smaller images
        w, h = img.size
        if w < self.hrCrop or h < self.hrCrop:
            img = img.resize(
                (max(w, self.hrCrop), max(h, self.hrCrop)), Resampling.BICUBIC
            )

        # Random Crop
        x = randint(0, w - self.hrCrop)
        y = randint(0, h - self.hrCrop)
        highRes = img.crop((x, y, x + self.hrCrop, y + self.hrCrop))

        lowResSize = self.hrCrop // self.scale
        lowRes = highRes.resize((lowResSize, lowResSize), Resampling.BICUBIC)

        return ImagePair(TF.to_tensor(lowRes), TF.to_tensor(highRes))
