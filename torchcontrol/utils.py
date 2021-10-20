import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


def smape(yhat, y):
    '''Add the mean'''
    return torch.abs(yhat - y) / (torch.abs(yhat) + torch.abs(y)) / 2


def mape(yhat, y):
    '''Add the mean'''
    return torch.abs((yhat - y)/yhat)


def get_macs(net:nn.Module, iters=1):
    params = []
    for p in net.parameters(): params.append(p.shape)
    with torch.cuda.device(0):
        macs, _ = get_model_complexity_info(net, (iters, params[0][1]), as_strings=False)
    return int(macs)


def get_flops(net:nn.Module, iters=1):
    # Relationship between macs and flops:
    # https://github.com/sovrasov/flops-counter.pytorch/issues/16
    params = []
    for p in net.parameters(): params.append(p.shape)
    with torch.cuda.device(0):
        macs, _ = get_model_complexity_info(net, (iters, params[0][1]), as_strings=False)
    return int(2*macs)

