"""

this file is tool for cwan network

"""
import torch
import torch.nn as nn
from mem import MemoryBlock,BNReLUConv,ReLUConv


class CWAN_L(nn.Module):
    """ 'L' of LAB's model.
    motivation
    ==========

    focus on enhancing image lightness and denoising

    name
    ====
    k -> kernel_size
    n -> output channels size
    x -> repeat number

    parameters
    =========
    k3n32 -> memory blocks. this block utilize local short skip connections whitin the bloack to represent short-term memory,as well as long skip connections sourcing from previous blocks to represent long-term memory.

    returns
    =======
    enhanced lightness image (1xHxW)

    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = ReLUConv(1,32)
        self.k3n1 = nn.Sequential(
                nn.Conv2d(32,1,(3,3),stride=1,padding=1)
        )
        self.memory_blocks = nn.ModuleList(
                [MemoryBlock(32,3,i+1) for i in range(3)]
        )

    def forward(self,l):
        residual = l
        out = self.feature_extractor(l)
        ys = [out]
        for memory_block in self.memory_blocks:
            out = memory_block(out,ys)
        out = self.k3n1(out)
        out = out + residual
        return out


class CWAN_AB(nn.Module):
    """ 'AB' of LAB's model.
    motivation
    ==========

    color infomation drive the attention of CWAN_AB

    name
    ====
    k -> kernel_size
    n -> output channels size

    parameters
    ==========


    returns
    =======
    1.enhanced color images(2xHxW)
    2.color attention maps(2xHxW)
    3.sparse attention points(2xHxW)

    """
    def __init__(self):
        super().__init__()

        self.k3n32_1 = nn.Sequential(
                nn.Conv2d(2,32,(3,3),stride=1,padding=1),
                nn.ReLU()
        )
        self.k3n32_2 = nn.Sequential(
                nn.Conv2d(4,32,(3,3),stride=1,padding=1),
                nn.ReLU()
        )

        k3n64_k1n128_k3n64 = nn.Sequential(
                nn.Conv2d(32,64,(3,3),stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(64,128,(1,1),stride=1,padding=0),
                nn.ReLU(),
                nn.Conv2d(128,64,(3,3),stride=1,padding=1),
                nn.ReLU()
        )
        self.k3n64_k1n128_k3n64_1 = nn.Sequential(
                nn.Conv2d(32,64,(3,3),stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(64,128,(1,1),stride=1,padding=0),
                nn.ReLU(),
                nn.Conv2d(128,64,(3,3),stride=1,padding=1),
                nn.ReLU()
        )
        self.k3n64_k1n128_k3n64_2 = nn.Sequential(
                nn.Conv2d(32,64,(3,3),stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(64,128,(1,1),stride=1,padding=0),
                nn.ReLU(),
                nn.Conv2d(128,64,(3,3),stride=1,padding=1),
                nn.ReLU()
        )
 
        self.k3n64_k1n128_k3n64_3 = nn.Sequential(
                nn.Conv2d(64,64,(3,3),stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(64,128,(1,1),stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(128,64,(3,3),stride=1,padding=1),
                nn.ReLU()
        )

        self.k3n2 = nn.Sequential(
                nn.Conv2d(64,2,(3,3),stride=1,padding=1)
        )
        self.k3n4 = nn.Sequential(
                nn.Conv2d(64,4,(3,3))
        )

    def forward(self,ab):
        residual = ab
        k3n32_1_output = self.k3n32_1(ab)
        k3n64_k1n128_k3n64_1_output = self.k3n64_k1n128_k3n64_1(k3n32_1_output)
        k3n2_output = self.k3n2(k3n64_k1n128_k3n64_1_output)
        attention_map = k3n2_output
        part2 = residual + k3n2_output
        cat_res_att = torch.cat([residual,k3n2_output],dim=1)
        k3n32_2_output = self.k3n32_2(cat_res_att)
        k3n64_k1n128_k3n64_2_output = self.k3n64_k1n128_k3n64_2(k3n32_2_output)
        k3n64_k1n128_k3n64_3_output = self.k3n64_k1n128_k3n64_3(k3n64_k1n128_k3n64_2_output)
        k3n4_output = self.k3n4(k3n64_k1n128_k3n64_3_output)
        attention_points = k3n4_output[:,2:]
        enhance_ab = residual + k3n4_output[:,:2]
        return enhance_ab,attention_map,attention_points

if __name__ == "__main__":
    print("Hello,cwan_parts.py!!!")

    cwan_l = CWAN_L()
    cwan_ab = CWAN_AB()
    rand_l = torch.rand(1,1,512,512)
    rand_ab = torch.rand(1,2,512,512)

    cwan_l_output = cwan_l(rand_l)
    print("CWAN_L models calculate...")
    print(cwan_l_output.shape)
    print("CWAN_AB models calculate...")
    cwan_ab_output,attention_map,attention_points = cwan_ab(rand_ab)
    print('cwan_ab_output.shape => {}'.format(cwan_ab_output.shape))
    print('attention_map.shape => {}'.format(attention_map.shape))
    print('attention_points.shape => {}'.format(attention_points.shape))
