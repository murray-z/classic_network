# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn

# 假设输入是 3*224*224

# 定义vgg架构(conv_layer_num, output_channel)，这里是VGG-11，卷积层数+全连接层数
vgg_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

# 重复n层卷积，输出通道数不变，最后使feature变小一半
def vgg_block(conv_layer_num, input_channels, output_channel):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(nn.Conv2d(input_channels, output_channel, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        input_channels = output_channel
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(input_channel, vgg_arch):
    conv_block = []
    for conv_layer_num, output_channel in vgg_arch:
        conv_block.append(vgg_block(conv_layer_num, input_channel, output_channel))
        input_channel = output_channel
    return nn.Sequential(*conv_block, nn.Flatten(),
                         nn.Linear(output_channel*7*7, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 1000))

vgg = vgg(3, vgg_arch)

x = torch.randn((5, 3, 224, 224))
for blk in vgg:
    x = blk(x)
    print(blk.__class__.__name__, "ouput shape\t", x.shape)



