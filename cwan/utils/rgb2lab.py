"""

this file is to decompose the low-light RGB to LAB color space(lightness and color components)

"""
import torch
import torch.nn as nn

class LAB():
    def __init__(self):
        super().init()

    def rgb2lab_tensor(self,rgb_tensor):
        pass
    def rgb2lab_rgb(self,rgb):
        pass

if __name__ == "__main__":
    print("Hello,rgb2lab.py!!!")
    lab = LAB()
