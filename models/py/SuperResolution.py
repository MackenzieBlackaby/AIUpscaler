import torch.nn as nn

"""
This file is the main Super Resolution CNN model
It is built of a single repeating convolutional block that calculates the residual
This modularity allows for on-the-fly customisation of the network's precision

"""


class ResidualBlock(nn.Module):
    """
    A single convolutional residual block for building the core of the neural network.
    """

    def __init__(self, channelCount: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channelCount, channelCount, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channelCount, channelCount, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """
    A single block for upsampling the image.
    """

    def __init__(self, features: int, scale: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features * (scale**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SupResNet(nn.Module):
    """
    This is the main Super Resolution neural network module
    """

    def __init__(
        self, scale: int = 2, channels: int = 3, features: int = 64, blockCount: int = 8
    ):
        super().__init__()
        assert scale in (2, 3, 4)
        self.scale = scale
        self.head = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(features) for _ in range(blockCount)])
        if scale in (2, 3):
            self.upsampleSection = nn.Sequential(UpsampleBlock(features, scale))
        else:
            self.upsampleSection = nn.Sequential(
                UpsampleBlock(features, 2), UpsampleBlock(features, 2)
            )
        self.tail = nn.Conv2d(features, channels, kernel_size=3, padding=1)

    def forward(self, x):
        base = nn.functional.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        feature = self.head(x)
        feature = self.body(feature)
        feature = self.upsampleSection(feature)
        residual = self.tail(feature)
        output = base + residual
        return output.clamp(0.0, 1.0)
