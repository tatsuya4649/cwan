"""

this file is to train CWAN model

"""

import torch
import torch.nn as nn
import argparse
from net.cwan_net import CWAN
from utils.rgb2lab import LAB
import glob
from tqdm import tqdm
from dataset.get_dataset import *
from dataset import get_dataset
from utils.test_tensor import Test
import math
import pickle
import os

parser = argparse.ArgumentParser(description="train cwan model")

parser.add_argument('-d','--dataset',help='dataset path',default="dataset/")
parser.add_argument('-lr','--learning_rate',help='learning_rate',default=1e-5,type=float)
parser.add_argument('-e','--epochs',help='number of epochs',default=200,type=int)
parser.add_argument('-b','--batch_size',help='batch size',default=16,type=int)
parser.add_argument('-wd','--weight_decay',default=0.05,type=float)
parser.add_argument('-mp','--model_path',default="models/")
parser.add_argument('--start_epoch',help="start epoch number",default=0,type=int)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('============================')
print('device is "{}"'.format(device))
print('============================')
args = parser.parse_args()
cwan = CWAN()
if args.start_epoch != 0:
    cwan.cwan_l.load_state_dict(torch.load("models/cwan_l_{}e.pth".format(args.start_epoch)))
cwan.train().to(device)
lab = LAB()
lab.eval().to(device)

optimizer = torch.optim.Adam([{'params':cwan.cwan_l.parameters()}],lr=args.learning_rate,weight_decay=args.weight_decay)
loss_func = nn.L1Loss()
"""

long_dict64 -> teaching data of RGB(dict)

short_dict64 -> data of RGB(list)

short_image -> imageid and patch of data of RGB(list)

"""
_BATCH = args.batch_size
_ONE_FILE_SIZE = get_dataset._ONE_FILE_SIZE
dataset = Dataset(64)
long_dic = dict()

if os.path.exists(args.model_path+"cwan_l_loss.pickle"):
    with open(args.model_path+"cwan_l_loss.pickle","rb") as file:
        loss_list = pickle.load(file)
else:
    loss_list = list()

_START_EPOCH = args.start_epoch + 1
test = Test("dark1.jpg","train_epoch_image/","../sample_images/","l",_START_EPOCH)
for e in tqdm(range(_START_EPOCH,args.epochs)):
    print("now {} epochs...".format(e))
    print("++++++++++++++++++++++++++++")
    while(not dataset.check_end):
        dataset.plus()#count+=1
        if dataset.change_now_check:
            print("loading ... long data(teaching data)")
            long_dic = dataset.long_dataset()
            print("setting longdata => long_dic !!!")
        print("short_imageid_list and short_list (data) is loading now...")
        print("----------------------now calculation pickle-----------------------")
        short_imageid_list,short_list = dataset.dataset_tensor()
        print("-------------------------------------------------------------------")
        for i in tqdm(range(math.ceil(float(_ONE_FILE_SIZE/_BATCH)))):#batch every time
            patch_tensor = Dataset.array_to_tensor(short_list[i*_BATCH:(i+1)*_BATCH]).to(device)
            patch_tensor = (patch_tensor.float()) / 255.
            patch_tensor_imageid = short_imageid_list[i*_BATCH:(i+1)*_BATCH]
            long_data = Dataset.search_long_data(long_dic,patch_tensor_imageid).to(device)
            long_data = (long_data.float()) / 255.
            patch_tensor_imageid = short_imageid_list[i*_BATCH:(i+1)*_BATCH]
            lab_long = lab(long_data)[:,:1,:,:]
            _,_,_,l_output,_ = cwan(patch_tensor)
            loss = loss_func(lab_long,l_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    dataset.count_reset()
    #check model generated
    cwan_l_output = cwan.l_test(test.im_tensor.to(device))
    loss_list.append(loss)
    test.tensor_image(cwan_l_output,loss_list)
    #save model parameters
    state_dict = cwan.cwan_l.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict,args.model_path+"cwan_l_{}e.pth".format(e))
    with open(args.model_path+"cwan_l_loss.pickle","wb") as file:
        pickle.dump(loss_list,file)
