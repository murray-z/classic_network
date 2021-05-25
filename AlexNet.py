# -*- coding: utf-8 -*-


import torch
import torch.nn as nn


# 假设输入是 3*224*224, 输出是1000
# x = 1*3*224*224

alexnet = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.Dropout(0.5),
    nn.Linear(4096, 4096), nn.Dropout(0.5),
    nn.Linear(4096, 1000)
)


x = torch.randn((5, 3, 224, 224))
for layer in alexnet:
    x = layer(x)
    print(layer.__class__.__name__, 'output shape: \t', x.shape)
