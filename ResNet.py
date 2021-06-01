# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_channels, use_1x1conv=False, strides=1):
        """
        残差单元: 每次通道数翻倍，图片大小减半
        :param in_channels: 输入通道数
        :param num_channels: 输出通道数
        :param use_1x1conv: 是否在残差中使用1x1卷积，主要是改变通道数
        :param strids: 步幅
        """
        super(Residual, self).__init__()
        # 输入输出的大小由strides决定
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # 1X1卷积
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def farward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return y


def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    """
    残差块
    :param in_channels:
    :param num_channels:
    :param num_residuals:
    :param first_block: 是否是第一个块，如果是则初始不让大小减半
    :return:
    """
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual(in_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            block.append(Residual(num_channels, num_channels))
    return block


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(
    *resnet_block(64, 64, 2, first_block=True)
)

b3 = nn.Sequential(
    *resnet_block(64, 128, 2)
)

b4 = nn.Sequential(
    *resnet_block(128, 256, 2)
)

b5 = nn.Sequential(
    *resnet_block(256, 512, 2)
)

# 最后输出采用均值池化，输出为通道数
resnet = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)
