import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import numpy as np

class FilterStripe(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(FilterStripe, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.ones(self.kernel_size*self.kernel_size*self.out_channels, self.in_channels,1,1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input):
        b, c, h, w = input.shape
        h = w = int(np.ceil(h / self.stride)) - 2*(self.kernel_size // 2 - self.padding)
        x = F.unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x = x.view(b, c, self.kernel_size * self.kernel_size, h, w).permute(0, 2, 1, 3, 4).reshape(b, c * self.kernel_size * self.kernel_size, h, w)
        x = F.conv2d(x, self.weight, groups=self.kernel_size * self.kernel_size)
        x = x.view(b, self.kernel_size * self.kernel_size, self.out_channels, h, w)
        print(x)#One can get stripe wise feature map for analization
        return torch.sum(x,dim=1)
    
    def load_standard_conv(self,weight):
        self.weight.data=weight.data.permute(2,3,0,1).reshape(self.out_channels*self.kernel_size*self.kernel_size,self.in_channels,1,1)
        
if __name__ == '__main__':
    conv=nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,padding=1,stride=2,bias=False)
    stripe=FilterStripe(in_channels=3,out_channels=2,kernel_size=3,stride=2)
    stripe.load_standard_conv(conv.weight)
    x = torch.rand(2, 3, 16, 16)
    print(stripe(x))
    print('stripe wise convolutional is equal to standard convolutional')
    print(conv(x))
    
    