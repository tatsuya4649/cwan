import torch
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

_DEFAULT_SIZE = 512

class Test:
    def __init__(self,name,train_images = "../train_epoch_image/",directories = "../../sample_images/",l_ab = "l",start_epoch=0):
        self._directories = directories
        self._train_images = train_images
        if not os.path.exists(self.train):
            print('not have train epoch image directories,make now')
            os.mkdir(self.train)#make _train_images directories
        self._path = self._directories + name
        self._tensor = self.image_tensor()
        self._epoch = start_epoch + 1
        self._name = name.split('.')[0]
        self._l_ab = l_ab

    @property
    def path(self):
        return self._path
    @property
    def name(self):
        return self._name
    @property
    def im_tensor(self):
        return self._tensor
    @property
    def epoch(self):
        return self._epoch
    @property
    def l_ab(self):
        return self._l_ab
    @property
    def directories(self):
        """
        directories have sample_images
        """
        return self._directories
    @property
    def train(self):
        """
        have train epoch image
        """
        return self._train_images
    @epoch.setter
    def epoch(self,x):
        self._epoch = x
    def image_tensor(self):
        """
        image path => PyTorch Tensor
        """
        im_numpy = cv2.imread(self._path)
        height = im_numpy.shape[0]
        width = im_numpy.shape[1]
        if width > height:
            im_numpy = cv2.resize(im_numpy,(_DEFAULT_SIZE,int(_DEFAULT_SIZE*(height/width)))) 
        else:
            im_numpy = cv2.resize(im_numpy,(int(_DEFAULT_SIZE*(width/height)),_DEFAULT_SIZE))
        im_tensor = torch.from_numpy(im_numpy).permute(2,0,1).unsqueeze(0).float()
        im_tensor /= 255.
        return im_tensor

    def tensor_image(self,tensor,loss_list,forward_epoch=True):
        """
        image path => PyTorch Tensor(BxCxHxW)
        """
        #only first batch element
        im_numpy = tensor[0].cpu().detach().numpy().transpose(1,2,0)#HxWxC
        before_numpy = self.im_tensor[0].cpu().detach().numpy().transpose(1,2,0)#HxWxC
        fig,ax = plt.subplots(figsize=(30,10),ncols=3)
        before = ax[0].imshow(before_numpy)
        after = ax[1].imshow(im_numpy)
        loss = ax[2].plot(np.array([ x+1 for x in range(len(loss_list))  ]),np.array(loss_list))
        fig.colorbar(before,ax=ax[0])
        fig.colorbar(after,ax=ax[1])
        plt.title("{} model training {} epochs".format(self.l_ab,self.epoch))
        plt.title("{} model training loss {} epochs".format(self.l_ab,self.epoch))
        plt.savefig("{}{}_{}e_{}.jpg".format(self.train,self.name,self.epoch,self.l_ab))
        if forward_epoch:
            self.epoch += 1



if __name__ == "__main__":
    image_path = "dark1.jpg"
    test = Test(image_path)
    rand = torch.rand(1,3,512,512)
    test.tensor_image(rand,False)
