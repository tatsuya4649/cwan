import torch


def loss_huber(target,input,size_average=None,delta=0.5,reduce=None,reduction='mean'):
    t = torch.abs(input-target)
    ret = torch.where(t <= delta,0.5*t**2,delta*t - (delta**2)/2)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret
