import torchvision.transforms.functional as TF

from random import randint
from PIL import Image
from PIL.Image import Resampling
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from kagglehub import dataset_download


def DownloadData(name: str = "daehoyang/flickr2k"):
    """
    Downloads the dataset with the given name. Will default to Flickr2K

    :param name: The name of the dataset to download
    :type name: str
    """
    path = dataset_download(name)
    return path


class TrainingDataImage:
    """
    Holds a high resolution and low resolution image pair as tensors for training
    """

    def __init__(self, lowRes: Tensor, highRes: Tensor):
        """
        Docstring for __init__

        :param self: self
        :param lowRes: Low resolution image tensor
        :type lowRes: Tensor
        :param highRes: High resolution image tensor
        :type highRes: Tensor
        """
        self.lowRes = lowRes
        self.highRes = highRes


class TrainingData(Dataset):
    """
    A dataset of images, inheriting from torch.utils.data.Dataset
    """

    def __init__(self, rootDir: str, scale: int = 2, hrCrop: int = 192):
        self.suffixes = {".jpg", ".png", ".jpeg", ".webp"}
        self.paths = sorted(
            [p for p in Path(rootDir).rglob("*") if p.suffix.lower() in self.suffixes]
        )
        self.scale = scale
        self.hrCrop = hrCrop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Returns a pair of low resolution and high resolution images

        :param self: self
        :param idx: Index of the image in the dataset
        """
        img = Image.open(self.paths[idx]).convert("RGB")

        w, h = img.size
        if w < self.hrCrop or h < self.hrCrop:
            img = img.resize(
                (max(w, self.hrCrop), max(h, self.hrCrop)), Resampling.BICUBIC
            )

        x = randint(0, w - self.hrCrop)
        y = randint(0, h - self.hrCrop)
        highRes = img.crop((x, y, x + self.hrCrop, y + self.hrCrop))

        lowResSize = self.hrCrop // self.scale
        lowRes = highRes.resize((lowResSize, lowResSize), Resampling.BICUBIC)

        return TrainingDataImage(TF.to_tensor(lowRes), TF.to_tensor(highRes))


# ==============================================

if __name__ == "__main__":
    DownloadData()
    pass

# ==============================================
