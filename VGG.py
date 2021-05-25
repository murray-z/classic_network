# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, conv_layer_num, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(conv_layer_num):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(DenseBlock, self).__init__()
        self.dense_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, out_size)
        )

    def forward(self, x):
        return self.dense_block(x)


class VGG(nn.Module):
    def __init__(self, in_channels, class_num, vgg_arch):
        super(VGG, self).__init__()
        conv_block = []
        for conv_layer_num, out_channels in vgg_arch:
            conv_block.append(ConvBlock(conv_layer_num, in_channels, out_channels))
            in_channels = out_channels
        self.conv_block = nn.Sequential(*conv_block)
        self.dense_block = DenseBlock(out_channels*7*7, class_num)

    def forward(self, x):
        x = self.conv_block(x)
        print(x.size())
        out = self.dense_block(x)
        return out


# 假设输入是 3*224*224
# 定义vgg架构(conv_layer_num, output_channel), 卷积层数+全连接层数
vgg_11_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
vgg_16_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
vgg_19_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))



vgg = VGG(3, 1000, vgg_16_arch)
print(vgg)
x = torch.randn((5, 3, 224, 224))
print(vgg(x))





