"""SRGAN discriminator."""

import torch.nn as nn


def _disc_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class SRGANDiscriminator(nn.Module):
    """Patch-like discriminator used for SRGAN."""

    def __init__(self, base_channels: int = 64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _disc_block(base_channels, base_channels, stride=2),
            _disc_block(base_channels, base_channels * 2, stride=1),
            _disc_block(base_channels * 2, base_channels * 2, stride=2),
            _disc_block(base_channels * 2, base_channels * 4, stride=1),
            _disc_block(base_channels * 4, base_channels * 4, stride=2),
            _disc_block(base_channels * 4, base_channels * 8, stride=1),
            _disc_block(base_channels * 8, base_channels * 8, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(base_channels * 8 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
