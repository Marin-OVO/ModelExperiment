"""conv of the u-net models"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLu(nn.Module):
    """
        Conv + Bn + ReLu
    """
    def __init__(self, in_ch: int, out_ch: int, mid_ch=None, kernel_size: int=3, dilation: int=1, with_bn: bool=True):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.with_bn = with_bn
        bias = not with_bn

        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, mid_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(mid_ch) if self.with_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.relu(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """
        double conv: conv + bn + ReLu x2
        Args:
            in_ch: in channels
            mid_ch: mid channels
            out_ch: out channels
    """
    def __init__(self, in_ch: int, out_ch: int, mid_ch=None, kernel_size: int=3, dilation: int=1, with_bn: bool=True):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch

        self.double_conv = nn.Sequential(
            ConvBNReLu(in_ch, mid_ch, kernel_size=kernel_size, dilation=dilation),
            ConvBNReLu(mid_ch, out_ch, kernel_size=kernel_size, dilation=dilation),
        )

    def forward(self, x):

        return self.double_conv(x)


class DownScaling(nn.Module):
    """
        downscaling of the u-net
        Args:
            in_ch: in channels
            out_ch: out channels
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # 2x2最大池化层
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):

        return self.maxpool_conv(x)


class UpScaling(nn.Module):
    """
    up conv and then double conv
    Args:
        c1: in channels
        c2: out channels
    """
    def __init__(self, c1, c2, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(c1, c2, c1 // 2)
        else:
            self.up = nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(c1, c2)

    def forward(self, x1, x2):
        """
            x1: output of the decoder, deep info
            x2: skip connection x1
        """
        x1 = self.up(x1)

        diffH = x2.size()[2] - x1.size()[2]
        diffW = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """
        1x1 conv, output prediction map
    """
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):

        return self.conv(x)

