"""

this file is to decompose the low-light RGB to LAB color space(lightness and color components)

"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
class LAB(nn.Module):
    def __init__(self):
        super().__init__()
    def _check_shape(self,tensor):
        print(tensor.shape)
        if tensor.shape[0] != 3:
            raise ValueError("Input array must have (batch, 3,height,width)")

    def rgb2xyz(self,rgb_tensor,show_results=False):
        """RGB to XYZ color space conversion.

        Parameters
        ==========

        rgb_tensor : shape -> (3,height,width) Tensor
        show_results : whether to display the resulting xyz image

        Returns
        ==========
        
        xyz_tensor : shape -> (3,height,width) Tensor

        what is xyz_tensor?
        -------------------
            -> https://www.dic-color.com/knowledge/xyz.html 

        """
        self._check_shape(rgb_tensor) #must have input shape {3,height,width}
        rgb_tensor = rgb_tensor.permute(1,2,0)
        mask = rgb_tensor > 0.04045
        rgb_tensor[mask] = torch.pow((rgb_tensor[mask] + 0.055)/1.055,2.4)
        rgb_tensor[~mask] /= 12.92
        xyz_from_rgb = torch.tensor([[0.412453, 0.357580, 0.180423],
                                     [0.212671, 0.715160, 0.072169],
                                     [0.019334, 0.119193, 0.950227]])
        xyz = torch.matmul(rgb_tensor,torch.t(xyz_from_rgb))
        if show_results: # show matplotlib
            xyz_numpy = xyz.cpu().detach().numpy()
            plt.imshow(xyz_numpy)
            plt.show()

        xyz = xyz.permute(2,0,1)
        return xyz

    def xyz2lab(self,xyz_tensor,show_results=False):
        """XYZ to CIE-LAB color space conversion.

        Parameters
        ==========

        xyz_tensor : shape -> (3,height,width) Tensor
        show_results : whether to display the resulting lab image
        
        Returns
        ==========
        
        lab_tensor : shape -> (3,height,width) Tensor

        what is lab_tensor?
        -------------------
            -> http://rysys.co.jp/dpex/help_laboutput.html 


        """
        xyz_tensor = xyz_tensor.permute(1,2,0)
        mask = xyz_tensor > 0.008856
        xyz_tensor[mask] = torch.pow(xyz_tensor[mask],1/3)
        xyz_tensor[~mask] = 7.787 * xyz_tensor[~mask] + 16. / 116.
        x,y,z = xyz_tensor[...,0],xyz_tensor[...,1],xyz_tensor[...,2]
        L = (116. * y) - 16.
        a = 500. * (x - y)
        b = 200. * (y - z)
        lab = torch.cat([L.unsqueeze(-1),a.unsqueeze(-1),b.unsqueeze(-1)],dim=-1)
        if show_results:
            lab_numpy = lab.cpu().detach().numpy()
            plt.imshow(lab_numpy)
            plt.show()

        lab = lab.permute(2,0,1)
        return lab


    def forward(self,rgb_tensor,show_xyz_results=False,show_lab_results=False):
        results = []
        for i in range(rgb_tensor.shape[0]):
            xyz = self.rgb2xyz(rgb_tensor[i],show_xyz_results)
            lab = self.xyz2lab(xyz,show_lab_results)
            results.append(lab)
        results = torch.cat(results).reshape(len(results),*results[0].shape)
        return results

if __name__ == "__main__":
    print("Hello,rgb2lab.py!!!")
    
    _IMAGE_PATH = "../../sample_images/silhouette-3038483_1920-1280x640.jpg"
    _DEFAULT_SIZE = 512
    im = cv2.imread(_IMAGE_PATH)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)#BGR->RGB
    height = im.shape[0]
    width = im.shape[1]
    print("image_height -> {}".format(height))
    print("image_width -> {}".format(width))
    im = im[height//2-_DEFAULT_SIZE//2:height//2+_DEFAULT_SIZE//2,width//2-_DEFAULT_SIZE//2:width//2+_DEFAULT_SIZE//2]
    lab = LAB()
    rgb_image = torch.from_numpy(im.transpose(2,0,1)).float()
    rgb_image = rgb_image.unsqueeze(0)
    rgb_image /= 255.
    lab_output = lab(rgb_image,True,True)

