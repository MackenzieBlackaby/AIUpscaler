import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.py.SuperResolution import SupResNet
from dataset.ImageSet import ImageSet, ImagePair
from dataset.DataDownloader import DownloadData


def main():
    trainImageDir = DownloadData()
    pass


if __name__ == "__main__":
    main()
