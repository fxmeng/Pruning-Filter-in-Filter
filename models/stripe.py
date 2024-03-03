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
        super(FilterStripe, self).__init__(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.BrokenTarget = None
        self.FilterSkeleton = Parameter(torch.ones(self.out_channels, self.kernel_size[0], self.kernel_size[1]), requires_grad=True)

    def forward(self, x):
        if self.BrokenTarget is not None:
            b, _, h, l = x.shape
            c, h_kernel, l_kernel = self.FilterSkeleton.shape
            self.BrokenTarget = self.BrokenTarget.to(x.device)
            x = F.conv2d(x, self.weight)
            zeros = torch.zeros(b, h_kernel*l_kernel*c, h, l, dtype=x.dtype, device=x.device)
            x = zeros.scatter_(1, self.BrokenTarget.expand(b, -1, h, l), x)
            x = x.view(b, h_kernel, l_kernel, c, h, l)
            out = torch.stack([self.shift(x[:,i, j], i, j) for i in range(h_kernel) for j in range(l_kernel)])
            out = out.sum(dim=0)[:, :, ::self.stride[0], ::self.stride[1]]
            return out
        else:
            return F.conv2d(x, self.weight * self.FilterSkeleton.unsqueeze(1), stride=self.stride, padding=self.padding)

    def prune_in(self, in_mask=None):
        self.weight = Parameter(self.weight[:, in_mask])
        self.in_channels = in_mask.sum().item()

    def prune_out(self, threshold):
        out_mask = (self.FilterSkeleton.abs() > threshold).sum(dim=(1, 2)) != 0
        self.weight = Parameter(self.weight[out_mask])
        self.FilterSkeleton = Parameter(self.FilterSkeleton[out_mask], requires_grad=True)
        self.out_channels = out_mask.sum().item()
        return out_mask

    def _break(self, threshold):
        self.BrokenTarget = self.FilterSkeleton.abs() > threshold
        self.out_channels = self.BrokenTarget.sum().item()
        self.BrokenTarget = self.BrokenTarget.permute(1, 2, 0).reshape(-1)
        self.weight = Parameter((self.weight * self.FilterSkeleton.unsqueeze(1)).permute(2, 3, 0, 1).reshape(-1, self.in_channels, 1, 1)[self.BrokenTarget].contiguous())
        self.BrokenTarget = torch.where(self.BrokenTarget)[0][None,:,None,None]
        self.kernel_size = (1, 1)

    def update_skeleton(self, sr, threshold):
        self.FilterSkeleton.grad.data.add_(sr * torch.sign(self.FilterSkeleton.data))
        mask = self.FilterSkeleton.data.abs() > threshold
        self.FilterSkeleton.data.mul_(mask)
        self.FilterSkeleton.grad.data.mul_(mask)
        out_mask = mask.sum(dim=(1, 2)) != 0
        return out_mask

    def shift(self, x, i, j):
        return F.pad(x, (self.FilterSkeleton.shape[1] // 2 - j, j - self.FilterSkeleton.shape[2] // 2, self.FilterSkeleton.shape[1] // 2 - i, i - self.FilterSkeleton.shape[2] // 2), 'constant', 0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
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
