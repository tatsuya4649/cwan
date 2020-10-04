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
from utils.loss_huber import loss_huber
from utils.attention_points import to_attention_points
from dataset.get_dataset import *
from dataset import get_dataset

parser = argparse.ArgumentParser(description="train cwan model")

parser.add_argument('-d','--dataset',help='dataset path',default="dataset/")
parser.add_argument('-lr','--learning_rate',help='learning_rate',default=1e-5)
parser.add_argument('-e','--epochs',help='number of epochs',default=200)
parser.add_argument('-b','--batch_size',help='batch size',default=32)
parser.add_argument('-a','--alpha',help='weights for alpha',default=1)
parser.add_argument('--beta',help='weights for beta',default=20)
parser.add_argument('-tl','--tau_l',help='weights for attention map minimum',default=0.05)
parser.add_argument('-tu','--tau_u',help='weights for attention map maximum',default=0.5)
parser.add_argument('-wd','--weight_decay',default=0.05)
parser.add_argument('-mp','--model_path',default="models/")
parser.add_argument('--l_path')
parser.add_argument('--delta',help='for huber loss',default=0.5)
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
print('======================')
print('device is "{}"'.format(device))
print('======================')


cwan = CWAN()
#loading cwan_l parameters
#cwan.cwan_l.load_state_dict(torch.load(args.l_path))
cwan.train()
lab = LAB()
lab.eval()

optimizer = torch.optim.Adam([{'params':cwan.cwan_ab.parameters()}],lr=args.learning_rate,weight_decay=args.weight_decay)
loss_func = nn.L1Loss()
loss_mse_func = nn.MSELoss()
"""

long_dict64 -> teaching data of RGB(dict)

short_dict64 -> data of RGB(list)

short_image -> imageid and patch of data of RGB(list)

"""
_BATCH = args.batch_size
_ONE_FILE_SIZE = get_dataset._ONE_FILE_SIZE
dataset = Dataset(32)
long_dic = dict()

test = Test("dark1.jpg","train_epoch_image/","../search_images/","ab")

for e in tqdm(range(args.epochs)):
    print("now {} epoch".format(e))
    while(True):
        dataset.plus()
        if dataset.change_now_check:
            print("loading ... long data and long attention map data (teaching data)")
            long_dic = dataset.long_dataset()
            long_attention_map = data.long_attention_map_dataset()
            print("setting long_data => long_dic and long_attention_map !!!")
        print('short_imageid_list and short_list(data) is loading now...')
        print('------------------- now calculation pickle ----------------')
        short_imageid_list,short_list = dataset.dataset_tensor()
        print('------------------------------------------------------------')
        for i in tqdm(range(int(_ONE_FILE_SIZE/_BATCH))):
            patch_tensor = Dataset.array_to_tensor(short_list[i*_BATCH:(i+1)*_BATCH])
            patch_tensor = (patch_tensor.float()) / 255.
            patch_tensor_imageid = short_imageid_list[i*_BATCH:(i+1)*_BATCH]
            long_data,long_attention_map = Dataset.long_data_search(long_dic,patch_tensor_imageid,long_attention_map)
            long_data = (long_data.float()) / 255.
            long_binary_points,long_attention_points = to_attention_points(long_attention_map)
            ab_long = lab(long_data)[:,1:3,:,:]
            _,attention_map,attentnion_points,_,ab_output = cwan(patch_tensor)
            #attention_maps loss
            loss_map = loss_func(attention_map,long_attention_map)
            #huber loss
            loss_huber = loss_huber(ab_output,long_ab)
            #mse loss
            loss_mse = (loss_mse_func(attention_points,long_attention_points) * Bp)/args.beta
            loss_ab = loss_huber + args.alpha * loss_mse
            #total loss
            loss = loss_map + loss_ab
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if dataset.check_end():
            dataset.count_reset()
            break
    #check model generated
    cwan_ab_output = cwan.ab_test(test.im_tensor)
    test.tensor_image(cwan_ab_output)
    #save model parameters
    state_dict = cwan.cwan_ab.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict,args.model_path+"cwan_ab_{}e.pth".format(e+1))
