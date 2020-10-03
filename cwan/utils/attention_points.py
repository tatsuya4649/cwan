import torch
import numpy as np
def to_attention_points(attention_maps,beta=20):
    """
    Parameters
    ==========
    attention_maps : of teaching data. shape=>BxCxHxW
    binary_point : shape => BxHxW
    """
    attention_points_list = []
    binary_points_list = []
    for n in range(attention_maps.shape[0]):
        binary_point = attention_maps[n,0,:,:]#use foreground color
        ones = torch.ones(binary_point.shape)
        zeros = torch.zeros(binary_point.shape)
        binary_point = torch.where(binary_point!=0,ones,zeros)
        one_count = len(binary_point[binary_point==1])
        if one_count > beta:
            np_binary = binary_point.cpu().detach().numpy()
            rand = np.random.randint(0,np.nonzero(np_binary)[0].shape[0]-1,beta)
            cat = np.stack([np.nonzero(np_binary)[0][rand],np.nonzero(np_binary)[1][rand]],axis=1)
            cat = cat.tolist()
            binary_point_np = np.zeros_like(binary_point)
            for i in cat:
                j,k = i
                binary_point_np[j][k] = 1
            binary_point = torch.from_numpy(binary_point_np)
        binary_points_list.append(binary_point)
        attention_points = torch.zeros(2,attention_maps.shape[-2],attention_maps.shape[-1])
        attention_points[0] = attention_maps[n,0,:,:]*binary_point 
        attention_points[1] = attention_maps[n,1,:,:]*binary_point
        attention_points_list.append(attention_points)
    attention_points = torch.cat(attention_points_list).reshape(len(attention_points_list),*attention_points_list[0].shape)
    binary_points = torch.cat(binary_points_list).reshape(len(binary_points_list),*binary_points_list[0].shape)
    return binary_points,attention_points
