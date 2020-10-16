import torch
import torch.nn as nn

class MemoryBlock(nn.Module):
    def __init__(self,channels,num_resblock,num_memblock):
        super().__init__()
        self.recursive_unit = nn.ModuleList(
                [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = ReLUConv((num_resblock+num_memblock)*channels,channels,1,1,0)
    def forward(self,x,ys):
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        gate_output = self.gate_unit(torch.cat(xs+ys,1))
        ys.append(gate_output)
        return gate_output

class ResidualBlock(nn.Module):
    def __init__(self,channels,k=3,s=1,p=1):
        super().__init__()
        self.relu_conv1 = ReLUConv(channels,channels,k,s,p)
        self.relu_conv2 = ReLUConv(channels,channels,k,s,p)
        self.relu_conv3 = ReLUConv(channels,channels,k,s,p)
    def forward(self,x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = self.relu_conv3(out)
        out = out + residual
        return out

class ReLUConv(nn.Sequential):
    def __init__(self,in_channels,channels,k=3,s=1,p=1,inplace=True):
        super().__init__()
        self.add_module('conv',nn.Conv2d(in_channels,channels,k,s,p,bias=False))
        self.add_module('relu',nn.ReLU(inplace=inplace))

class BNReLUConv(nn.Sequential):
    def __init__(self,in_channels,channels,k=3,s=1,p=1,inplace=True):
        super().__init__()
        self.add_module('bn',nn.BatchNorm2d(in_channels))
        self.add_module('relu',nn.ReLU(inplace=inplace))
        self.add_module('conv',nn.Conv2d(in_channels,channels,k,s,p,bias=False))
