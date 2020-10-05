"""
this file is to make cwan network of deep learning.
"""
import torch
import torch.nn as nn
import sys,os

sys.path.append(os.path.abspath(".."))
from utils.rgb2lab import LAB
from .cwan_parts import CWAN_L,CWAN_AB

class CWAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cwan_l = CWAN_L()
        self.cwan_ab = CWAN_AB()
        self.lab_converter = LAB()
    def lab2rgb(self,lab):
        """ LAB tensor to RGB tensor

        Parameter
        =========

        lab : LAB image tensor

        Returns
        =========

        rgb : RGB image tensor

        """
        rgb = self.lab_converter.lab2rgb(lab)
        return rgb

    def l_test(self,tensor):
        lab = self.lab_converter(tensor)
        l = lab[:,:1]
        l_output = self.cwan_l(l)
        return l_output

    def ab_test(self,tensor):
        lab = self.lab_converter(tensor)
        ab = lab[:,1]
        ab_output,_,_ = self.cwan_ab(ab)
        return ab_output

    def forward(self,tensor):
        lab = self.lab_converter(tensor)
        l,ab = lab[:,:1],lab[:,1:]
        l_output = self.cwan_l(l)
        ab_output,attention_map,attention_points = self.cwan_ab(ab)
        generated_image = torch.rand(tensor.shape)
        generated_image[:,0] = l_output[:,0]
        generated_image[:,1] = ab_output[:,0]
        generated_image[:,2] = ab_output[:,1]
        return generated_image,attention_map,attention_points,l_output,ab_output


if __name__ == "__main__":
    cwan = CWAN()
    rand = torch.rand(1,3,512,512)
    cwan_output,attention_map,attention_points,_,_ = cwan(rand)
    print(cwan_output.shape)

