# -*- coding: UTF-8 -*-
# network in network

import torch
import torch.nn as nn
import torch.nn.functional as F


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1),   # 卷积后接两层1X1卷积，增加通道非线性
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1),
        nn.ReLU()
    )


# 主要框架和AlexNet类似
nin_net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5), # ?
    # 设置标签数为最后的通道数(这里假设10类)
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    # 通过平均池化，生成输出维度
    nn.AdaptiveAvgPool2d((1, 1)),
    # 使输出维度为  batch_size X class_num
    nn.Flatten()
)


x = torch.randn((5, 1, 224, 224))
for layer in nin_net:
    x = layer(x)
    print(layer.__class__.__name__, 'output shape: \t', x.shape)
