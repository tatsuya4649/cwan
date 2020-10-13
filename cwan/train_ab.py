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
from utils.test_tensor import Test
import math
import pickle

parser = argparse.ArgumentParser(description="train cwan model")

parser.add_argument('-d','--dataset',help='dataset path',default="dataset/")
parser.add_argument('-lr','--learning_rate',help='learning_rate',default=1e-5)
parser.add_argument('-e','--epochs',help='number of epochs',default=200)
parser.add_argument('-b','--batch_size',help='batch size',default=128)
parser.add_argument('-a','--alpha',help='weights for alpha',default=1)
parser.add_argument('--beta',help='weights for beta',default=20)
parser.add_argument('-tl','--tau_l',help='weights for attention map minimum',default=0.05)
parser.add_argument('-tu','--tau_u',help='weights for attention map maximum',default=0.5)
parser.add_argument('-wd','--weight_decay',default=0.05)
parser.add_argument('-mp','--model_path',default="models/")
parser.add_argument('--l_path',default="models/cwan_l_{}e.pth".format(83))
parser.add_argument('--delta',help='for huber loss',default=0.5)
parser.add_argument('-s','--start_epoch',help='start epoch count',default=0,type=int)
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
print('======================')
print('device is "{}"'.format(device))
print('======================')


cwan = CWAN()
#loading cwan_l parameters
cwan.cwan_l.load_state_dict(torch.load(args.l_path))
cwan.train().to(device)
lab = LAB()
lab.eval().to(device)

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

if os.path.exists(args.model_path+"cwan_ab_loss.pickle"):
    with open(args.model_path + "cwan_ab_loss.pickle","rb") as file:
        loss_list = pickle.load(file)
else:
    loss_list = list()

if os.path.exists(args.model_path+"cwan_ab_loss_map.pickle"):
    with open(args.model_path + "cwan_ab_loss_map.pickle","rb") as file:
        loss_map_list = pickle.load(file)
else:
    loss_map_list = list()

if os.path.exists(args.model_path+"cwan_ab_loss_huber.pickle"):
    with open(args.model_path + "cwan_ab_loss_huber.pickle","rb") as file:
        loss_huber_list = pickle.load(file)
else:
    loss_huber_list = list()

if os.path.exists(args.model_path+"cwan_ab_loss_mse.pickle"):
    with open(args.model_path + "cwan_ab_loss_mse.pickle","rb") as file:
        loss_mse_list = pickle.load(file)
else:
    loss_mse_list = list()

_START_EPOCH = args.start_epoch + 1
test = Test("dark1.jpg","train_epoch_image/","../sample_images/","ab",_START_EPOCH)

for e in tqdm(range(_START_EPOCH,args.epochs)):
    print("now {} epoch".format(e))
    print("+++++++++++++++++++++++++")
    while(not dataset.check_end):
        dataset.plus()
        if dataset.change_now_check:
            print("loading ... long data and long attention map data (teaching data)")
            long_dic = dataset.long_dataset()
            long_attention_map_dic = dataset.long_attention_map_dataset()
            print("setting long_data => long_dic and long_attention_map !!!")
        print('short_imageid_list and short_list(data) is loading now...')
        print('------------------- now calculation pickle ----------------')
        short_imageid_list,short_list = dataset.dataset_tensor()
        print('------------------------------------------------------------')
        for i in tqdm(range(math.ceil(float(_ONE_FILE_SIZE/_BATCH)))):
            patch_tensor = Dataset.array_to_tensor(short_list[i*_BATCH:(i+1)*_BATCH]).to(device)
            patch_tensor = (patch_tensor.float()) / 255.
            patch_tensor_imageid = short_imageid_list[i*_BATCH:(i+1)*_BATCH]
            long_data,long_attention_map = Dataset.search_long_data(long_dic,patch_tensor_imageid,long_attention_map_dic)
            long_data = long_data.to(device)
            long_attention_map = long_attention_map.to(device)
            long_data = (long_data.float()) / 255.
            long_binary_points,long_attention_points = to_attention_points(long_attention_map)
            ab_long = lab(long_data)[:,1:3,:,:]
            _,attention_map,attention_points,_,ab_output = cwan(patch_tensor)
            #attention_maps loss
            loss_map = loss_func(attention_map,long_attention_map)
            #huber loss
            loss_huber_output = loss_huber(ab_output,ab_long)
            #mse loss
            long_binary_points = long_binary_points.unsqueeze(1).to(device)
            long_attention_points = long_attention_points.to(device)
            loss_mse = loss_mse_func(attention_points*long_binary_points ,long_attention_points*long_binary_points)/float(args.beta)
            loss_ab = loss_huber_output + args.alpha * loss_mse
            #total loss
            loss = loss_map + loss_ab
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    dataset.count_reset()
    #check model generated
    cwan_ab_output = cwan.ab_test(test.im_tensor.to(device))
    loss_map_list.append(loss_map)
    loss_huber_list.append(loss_huber)
    loss_mse_list.append(loss_mse)
    loss_list.append(loss)
    test.tensor_image(cwan_ab_output,loss_list)
    #save model parameters
    state_dict = cwan.cwan_ab.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict,args.model_path+"cwan_ab_{}e.pth".format(e))
    with open(args.model_path + "cwan_ab_loss.pickle","wb") as file:
        pickle.dump(loss_list,file)
    with open(args.model_path + "cwan_ab_loss_map.pickle","wb") as file:
        pickle.dump(loss_map_list,file)
    with open(args.model_path + "cwan_ab_loss_huber.pickle","wb") as file:
        pickle.dump(loss_huber_list,file)
    with open(args.model_path + "cwan_ab_loss_mse.pickle","wb") as file:
        pickle.dump(loss_mse_list,file)
    
