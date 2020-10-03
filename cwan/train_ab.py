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
patch_tensor = torch.rand(1,3,64,64)
long_data = torch.rand(1,3,64,64)
long_attention_map = torch.rand(1,2,64,64)
long_binary_points,long_attention_points = to_attention_points(long_attention_map)
ab_long = lab(long_data)[:,1:3,:,:]
print(ab_long.shape)
_,attention_map,attention_points,_,ab_output = cwan(patch_tensor)
#attention_maps loss
loss_map = loss_func(attention_map,long_attention_map)
#huber loss
loss_huber = loss_huber(ab_output,ab_long)
#mse loss
loss_mse = (loss_mse_func(attention_points,long_attention_points) * long_binary_points)/args.beta
loss_ab = loss_huber + args.alpha * loss_mse
#total loss
loss = loss_map + loss_ab
raise ValueError("a")
_BATCH = args.batch_size
for e in tqdm(range(args.epochs)):
    print("now {} epoch".format(e))
    patch_tensor = torch.rand(1,3,64,64)
    long_data = torch.rand(1,3,64,64)
    long_attention_map = torch.rand(1,2,64,64)
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

    #save model parameters
    state_dict = cwan.cwan_ab.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict,args.model_path+"cwan_ab_{}.pth".format(e))
