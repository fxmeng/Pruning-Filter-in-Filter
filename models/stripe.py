import torch.nn as nn
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

__all__ = ['FilterStripe', 'BatchNorm', 'Linear']


class FilterStripe(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(FilterStripe, self).__init__(in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=1, bias=False)
        self.BrokenTarget = None
        self.FilterSkeleton = Parameter(torch.ones(self.out_channels, self.kernel_size[0], self.kernel_size[1]), requires_grad=True)

    def forward(self, x):
        if self.BrokenTarget is not None:
            out = torch.zeros(x.shape[0], self.FilterSkeleton.shape[0], int(np.ceil(x.shape[2] / self.stride[0])), int(np.ceil(x.shape[3] / self.stride[1])))
            if x.is_cuda:
                out = out.cuda()
            x = F.conv2d(x, self.weight)
            l, h = 0, 0
            for i in range(self.BrokenTarget.shape[0]):
                for j in range(self.BrokenTarget.shape[1]):
                    h += self.FilterSkeleton[:, i, j].sum().item()
                    out[:, self.FilterSkeleton[:, i, j]] += self.shift(x[:, l:h], i, j)[:, :, ::self.stride[0], ::self.stride[1]]
                    l += self.FilterSkeleton[:, i, j].sum().item()
            return out
        else:
            return F.conv2d(x, self.weight * self.FilterSkeleton.unsqueeze(1), stride=self.stride, padding=self.padding, groups=self.groups)

    def prune_in(self, in_mask=None):
        self.weight = Parameter(self.weight[:, in_mask])
        self.in_channels = in_mask.sum().item()

    def prune_out(self, threshold):
        out_mask = (self.FilterSkeleton.abs() > threshold).sum(dim=(1, 2)) != 0
        if out_mask.sum() == 0:
            out_mask[0] = True
        self.weight = Parameter(self.weight[out_mask])
        self.FilterSkeleton = Parameter(self.FilterSkeleton[out_mask], requires_grad=True)
        self.out_channels = out_mask.sum().item()
        return out_mask

    def _break(self, threshold):
        self.weight = Parameter(self.weight * self.FilterSkeleton.unsqueeze(1))
        self.FilterSkeleton = Parameter((self.FilterSkeleton.abs() > threshold), requires_grad=False)
        if self.FilterSkeleton.sum() == 0:
            self.FilterSkeleton.data[0][0][0] = True
        self.out_channels = self.FilterSkeleton.sum().item()
        self.BrokenTarget = self.FilterSkeleton.sum(dim=0)
        self.kernel_size = (1, 1)
        self.weight = Parameter(self.weight.permute(2, 3, 0, 1).reshape(-1, self.in_channels, 1, 1)[self.FilterSkeleton.permute(1, 2, 0).reshape(-1)])

    def update_skeleton(self, sr, threshold):
        self.FilterSkeleton.grad.data.add_(sr * torch.sign(self.FilterSkeleton.data))
        mask = self.FilterSkeleton.data.abs() > threshold
        self.FilterSkeleton.data.mul_(mask)
        self.FilterSkeleton.grad.data.mul_(mask)
        out_mask = mask.sum(dim=(1, 2)) != 0
        return out_mask

    def shift(self, x, i, j):
        return F.pad(x, (self.BrokenTarget.shape[0] // 2 - j, j - self.BrokenTarget.shape[0] // 2, self.BrokenTarget.shape[0] // 2 - i, i - self.BrokenTarget.shape[1] // 2), 'constant', 0)

    def extra_repr(self):
        s = ('{BrokenTarget},{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        return s.format(**self.__dict__)


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__(num_features)
        self.weight.data.fill_(0.5)

    def prune(self, mask=None):
        self.weight = Parameter(self.weight[mask])
        self.bias = Parameter(self.bias[mask])
        self.register_buffer('running_mean', self.running_mean[mask])
        self.register_buffer('running_var', self.running_var[mask])
        self.num_features = mask.sum().item()

    def update_mask(self, mask=None, threshold=None):
        if mask is None:
            mask = self.weight.data.abs() > threshold
        self.weight.data.mul_(mask)
        self.bias.data.mul_(mask)
        self.weight.grad.data.mul_(mask)
        self.bias.grad.data.mul_(mask)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__(in_features, out_features)
        self.weight.data.normal_(0, 0.01)

    def prune_in(self, mask=None):
        self.in_features = mask.sum().item()
        self.weight = Parameter(self.weight[:, mask])

    def prune_out(self, mask=None):
        self.out_features = mask.sum().item()
        self.weight = Parameter(self.weight[mask])
        self.bias = Parameter(self.bias[mask])
