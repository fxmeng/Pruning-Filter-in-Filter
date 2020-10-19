import torch.nn as nn
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from .stripe import *

class Stripe_Group(nn.Conv2d):
    def __init__(self, in_channels, stride=1):
        super(Stripe_Group, self).__init__(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.mask = Parameter(torch.ones_like(self.weight))
    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, stride=self.stride, padding=1, groups=self.groups)
    def update_skeleton(self, sr, threshold):
        self.mask.grad.data.add_(sr * torch.sign(self.mask.data))
        mask = self.mask.data.abs() > threshold
        self.mask.data.mul_(mask)
        self.mask.grad.data.mul_(mask)
        out_mask = mask.sum(dim=(1, 2, 3)) != 0
        return out_mask

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = FilterStripe(in_planes, planes, kernel_size=1, stride=1)
        self.bn1 = BatchNorm(planes)
        self.conv2 = Stripe_Group(planes,stride=stride)
        self.bn2 = BatchNorm(planes)
        self.conv3 = FilterStripe(planes, out_planes, kernel_size=1, stride=1)
        self.bn3 = BatchNorm(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                FilterStripe(in_planes, out_planes, kernel_size=1, stride=1),
                BatchNorm(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = FilterStripe(3, 32, kernel_size=3, stride=1)
        self.bn1 = BatchNorm(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = FilterStripe(320, 1280, kernel_size=1, stride=1)
        self.bn2 = BatchNorm(1280)
        self.linear = Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out