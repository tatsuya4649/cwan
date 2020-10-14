"""

this file is to test CWAN model.

"""

import torch
import cv2
import argparse
from net.cwan_net import CWAN
from matplotlib import pyplot as plt
from utils.rgb2lab import LAB
import sys
sys.path.append('net')

parser = argparse.ArgumentParser(description='image -> CWAN model -> LLIE image')
#image path
parser.add_argument('-i','--image',help='test image name',default='silhouette-3038483_1920-1280x640.jpg')
#whether show xyz image
parser.add_argument('-x','--xyz',help='show xyz image',default=False)
#whether show lab image
parser.add_argument('-l','--lab',help='show lab image',default=False)
#whether show results image
parser.add_argument('-r','--results',help='show resutls image',default=True)
#cwan model state path
parser.add_argument('-m','--model_state',help='cwan pytorch model state path')
#test image default size
parser.add_argument('-s','--default_size',help='test image default size',default=512)
#can_l model
parser.add_argument('--cwan_l',help='cwan_l model parameters epoch number',default=83)
parser.add_argument('--only_l',help='test only cwan_l',default=False)
args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("======================")
    print("device => {}".format(device))
    print("======================")
    _DEFAULT_SIZE = args.default_size
    _IMAGE_PATH = "../sample_images/{}".format(args.image)
    _MODEL_STATE_PATH = "models/{}".format(args.model_state)
    _L_MODEL_STATE_PATH = "models/cwan_l_{}e.pth".format(args.cwan_l)
    test_image = cv2.imread(_IMAGE_PATH)
    test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
    height = test_image.shape[0]
    width = test_image.shape[1]
    print("test image size...{}x{}".format(height,width))

    test_image = test_image[height//2-_DEFAULT_SIZE//2:height//2+_DEFAULT_SIZE//2,width//2-_DEFAULT_SIZE//2:width//2+_DEFAULT_SIZE//2]
    test_tensor = torch.from_numpy(test_image).float()
    test_tensor /= 255.
    test_tensor = test_tensor.permute(2,0,1)
    test_tensor = test_tensor.unsqueeze(0).to(device)
    #============= end of ready image =============
    print("now predicting image...")
    cwan = CWAN().eval().to(device)
    if args.cwan_l is not None:
        print("loading CWAN_L model parameters....")
        cwan.cwan_l.load_state_dict(torch.load(_L_MODEL_STATE_PATH))
    if args.model_state is not None:# load pre-trained mode
        cwan.load_state_dict(torch.load(_MODEL_STATE_PATH))
    if args.only_l:#only test cwan_L model
        print("predict only CWAN_L model")
        l_output = cwan.l_test(test_tensor)
        print(l_output)
        print(l_output.shape)
        if args.results:
            print('+++++++++++++++++++++')
            print('show results !!!')
            lab = LAB()
            _l_before = lab(test_tensor)
            _l_before = _l_before[:,:1]
            fig = plt.figure(figsize=(20,10))
            before = plt.subplot(1,2,1)
            after = plt.subplot(1,2,2)
            before.imshow(_l_before[0].cpu().detach().numpy().transpose(1,2,0),cmap="gray")
            after.imshow(l_output[0].cpu().detach().numpy().transpose(1,2,0),cmap="gray") 
            plt.show()
    else:
        cwan_output,_,_,_,_ = cwan(test_tensor)
        #============= lab -> rgb =====================
        rgb_output = cwan.lab2rgb(cwan_output)
        if args.results:
            plt.imshow(rgb_output[0].cpu().detach().numpy().transpose(1,2,0))
            plt.show()
