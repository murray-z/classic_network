# -*- coding: utf-8 -*-


import torch
import torch.nn as nn


# 假设输入为1*32*32; 输出是10

lenet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=(5, 5)), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(6, 16, kernel_size=(5, 5)), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2), nn.Flatten(),
    nn.Linear(16*5*5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)


x = torch.randn((5, 1, 32, 32))
for layer in lenet:
    x = layer(x)
    print(layer.__class__.__name__, 'output shape: \t', x.shape)