"""BiFPN模块
Bidirectional Feature Pyramid Network
参考: https://arxiv.org/abs/1911.09070
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BiFPNBlock(nn.Module):
    def __init__(self, num_channels=256):
        super(BiFPNBlock, self).__init__()
        self.num_channels = num_channels

        self.conv6_up = ConvBlock(num_channels, num_channels)
        self.conv5_up = ConvBlock(num_channels, num_channels)
        self.conv4_up = ConvBlock(num_channels, num_channels)

        self.conv4_down = ConvBlock(num_channels, num_channels)
        self.conv5_down = ConvBlock(num_channels, num_channels)
        self.conv6_down = ConvBlock(num_channels, num_channels)
        self.conv7_down = ConvBlock(num_channels, num_channels)

    def forward(self, inputs):
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        p7_up = p7_in
        p6_up = self.conv6_up(p6_in + F.interpolate(p7_up, size=p6_in.shape[-2:], mode='nearest'))
        p5_up = self.conv5_up(p5_in + F.interpolate(p6_up, size=p5_in.shape[-2:], mode='nearest'))
        p4_up = self.conv4_up(p4_in + F.interpolate(p5_up, size=p4_in.shape[-2:], mode='nearest'))

        p3_out = p3_in
        p4_out = self.conv4_down(p4_up)
        p5_out = self.conv5_down(p5_up + F.interpolate(p4_out, size=p5_up.shape[-2:], mode='nearest'))
        p6_out = self.conv6_down(p6_up + F.interpolate(p5_out, size=p6_up.shape[-2:], mode='nearest'))
        p7_out = self.conv7_down(p7_up + F.interpolate(p6_out, size=p7_up.shape[-2:], mode='nearest'))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, num_channels=256, num_repeats=3):
        super(BiFPN, self).__init__()
        self.num_channels = num_channels
        self.num_repeats = num_repeats

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, num_channels, 1) for in_ch in in_channels_list
        ])

        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(num_channels) for _ in range(num_repeats)
        ])

    def forward(self, features):
        features = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for block in self.bifpn_blocks:
            features = block(features)

        return features
