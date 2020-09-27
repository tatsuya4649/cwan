"""

this file is to test CWAN model.

"""

import torch
import cv2
import argparse
from net.cwan_net import CWAN
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='image -> CWAN model -> LLIE image')
#image path
parser.add_argument('-i','--image',help='test image name',default='silhouette-3038483_1920-1280x640.jpg')
#whether show xyz image
parser.add_argument('-x','--xyz',help='show xyz image',default=False)
#whether show lab image
parser.add_argument('-l','--lab',help='show lab image',default=False)
#whether show results image
parser.add_argument('-r','--results',help='show resutls image',default=True)
args = parser.parse_args()

if __name__ == "__main__":
    _DEFAULT_SIZE = 512
    _IMAGE_PATH = "../sample_images/{}".format(args.image)
    test_image = cv2.imread(_IMAGE_PATH)
    test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
    height = test_image.shape[0]
    width = test_image.shape[1]
    print("test image size...{}x{}".format(height,width))

    test_image = test_image[height//2-_DEFAULT_SIZE//2:height//2+_DEFAULT_SIZE//2,width//2-_DEFAULT_SIZE//2:width//2+_DEFAULT_SIZE//2]
    test_tensor = torch.from_numpy(test_image).float()
    test_tensor /= 255.
    test_tensor = test_tensor.permute(2,0,1)
    test_tensor = test_tensor.unsqueeze(0)
    #============= end of ready image =============
    cwan = CWAN()
    cwan_output,_,_ = cwan(test_tensor)
    print(cwan_output.shape)
    if args.results:
        plt.imshow(cwan_output[0].cpu().detach().numpy().transpose(1,2,0))
        plt.show()
