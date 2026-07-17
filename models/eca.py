"""ECA注意力模块
Efficient Channel Attention (ECA)
参考: https://arxiv.org/abs/1910.03151
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ECA, self).__init__()
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
