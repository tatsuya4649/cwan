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

parser = argparse.ArgumentParser(description="train cwan model")

parser.add_argument('-d','--dataset',help='dataset path',default="dataset/")
parser.add_argument('-lr','--learning_rate',help='learning_rate',default=1e-5)
parser.add_argument('-e','--epochs',help='number of epochs',default=200)
parser.add_argument('-b','--batch_size',help='batch size',default=16)
parser.add_argument('-wd','--weight_decay',default=0.05)
parser.add_argument('-mp','--model_path',default="models/")

args = parser.parse_args()
cwan = CWAN()
cwan.train()
lab = LAB()
lab.eval()

optimizer = torch.optim.Adam([{'params':cwan.cwan_l.parameters()}],lr=args.learning_rate,weight_decay=args.weight_decay)
loss_func = nn.L1Loss()
"""

long_dict64 -> teaching data of RGB(dict)

short_dict64 -> data of RGB(list)

short_image -> imageid and patch of data of RGB(list)

"""
_BATCH = args.batch_size
dataset = Dataset(64)
long_dic = dict()
for e in tqdm(range(args.epochs)):
    print("now {} epochs...".format(e))
    while(True):
        dataset.plus()#count+=1
        if dataset.change_now_check:
            long_dic = dataset.long_dataset()
        short_imageid_list,short_list = dataset.dataset_tensor()
        for i in range(_ONE_FILE_SIZE/_BATCH):#batch every time
            patch_tensor = short_list[i*_BATCH:(i+1)*_BATCH]
            long_data = torch.randint(1,3,64,64)
            lab_long = lab(long_data)[:,:1,:,:]
            _,_,_,l_output,_ = cwan(patch_tensor)
            loss = loss_func(long_data,l_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if dataset.check_end():
            break
    #save model parameters
    state_dict = cwan.cwan_l.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict,args.model_path+"cwan_l_{}.pth".format(e))