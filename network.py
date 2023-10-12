import os
import torch
import numpy as np
import time
from torch import nn
from eval import *
from collections import OrderedDict
from torchmeta.modules import MetaModule
from torch.nn.parameter import Parameter

""" 激活函数 """
class Sine(nn.Module):
    """ 初始化 """
    def __init__(self, w0=30., activate=True):
        super().__init__()
        self.w0 = w0
        self.activate = activate

    """ 前向传播 """
    def forward(self, x):
        if self.activate:
            return torch.sin(self.w0*x)
        else:
            return x

""" 线性层 """
class Linear(nn.Linear, MetaModule):
    """ 前向传播 """
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params["bias"]
        weight = params["weight"]
        output = input.matmul(weight.permute(-1, -2)) + bias
        return output

""" COIN, SW 网络 """
class SW(MetaModule):
    """ 初始化 """
    def __init__(self, args, in_feats, hid_feats, out_feats, hid_num, w0=30.):
        super().__init__()
        self.args = args
        self.c_width, self.c_depth = hid_feats, hid_num
        self.net = []
        self.plot_hist = True if self.args.state == "demo" else False
        self.res_list = []
        self.net.append(nn.Sequential(Linear(in_feats, hid_feats), Sine(w0)))
        for _ in range(hid_num):
            self.net.append(nn.Sequential(Linear(hid_feats, hid_feats), Sine(w0)))
        self.net.append(nn.Sequential(Linear(hid_feats, out_feats), Sine(activate=False)))
        self.net = nn.Sequential(*self.net)
        with torch.no_grad():
            for i, layer in enumerate(self.net):
                init_weight(layer[0].weight, w0, i==0)
                init_bias(layer[0].bias)

    """ 前向传播 """
    def forward(self, input, params=None):
        output = input  
        for i, layer in enumerate(self.net):
            output = layer[0](output, params=self.get_subdict(params, f"net.{i}.0"))
            output = layer[1](output)
        return output

    def load_net(self, net, s_width):
        left, all = slice(None, s_width), slice(None)
        with torch.no_grad():
            w_tmp, b_tmp = [], []
            for i, layer in enumerate(net.net):
                w_tmp.append(layer[0].weight)
                b_tmp.append(layer[0].bias)
            for i, layer in enumerate(self.net):
                if (i==0):
                    layer[0].weight[left, all] = w_tmp[i][left, all]
                    layer[0].bias[left] = b_tmp[i][left]
                elif (i==len(self.net)-1):
                    layer[0].weight[all, left] = w_tmp[i][all, left]
                    layer[0].bias[all] = b_tmp[i][all]
                else:
                    layer[0].weight[left, left] = w_tmp[i][left, left]
                    layer[0].bias[left] = b_tmp[i][left]

    def freeze_net(self, s_width):
        left, all = slice(None, s_width), slice(None)
        for i, layer in enumerate(self.net):
            if (i==len(self.net)-1):
                layer[0].weight.grad[all, left] = 0
                layer[0].bias.grad[all] = 0
            else:
                layer[0].weight.grad[left, all] = 0
                layer[0].bias.grad[left] = 0

    def prune_net(self, s_width, prune_inner=True, prune_out=True):
        left, right, all = slice(None, s_width), slice(s_width, None), slice(None)
        with torch.no_grad():
            for i, layer in enumerate(self.net):
                if (i!=0 and i!=(len(self.net)-1)):
                    layer[0].weight[left, right] = 0
                    if prune_inner:
                        layer[0].weight[right, right] = 0
            if prune_out:
                self.net[-1][0].weight[all, right] = 0

class SWD(MetaModule):
    """ 初始化 """
    def __init__(self, args, in_feats, hid_feats, out_feats, hid_num, w0=30.):
        super().__init__()
        self.args = args
        self.state = args.state
        self.plot_distribution = args.plot_distribution
        self.depths = list(set(args.depths))               ## [0:2, 1:2, 2:4]
        (self.depths).sort()
        self.out_idx = {}
        for i, depth in enumerate(self.depths):         
            self.out_idx[depth] = i                 ## {2:0, 4:1, 6:2}
        self.c_width, self.c_depth = hid_feats, hid_num
        self.mode_path = os.path.join(self.args.logs_path, f"{args.mode}")
        self.info_path = os.path.join(self.mode_path, "info/")
        self.mods_train_path = os.path.join(self.mode_path, "mods/", "train/")
        self.imgs_train_path = os.path.join(self.mode_path, "imgs/", "train/")

        self.in_net, self.hid_net, self.out_net = [], [], []
        self.in_net.append(nn.Sequential(Linear(in_feats, hid_feats), Sine(w0)))
        self.in_net = nn.Sequential(*self.in_net)
        for _ in range(hid_num):
            self.hid_net.append(nn.Sequential(Linear(hid_feats, hid_feats), Sine(w0)))   
        self.hid_net = nn.Sequential(*self.hid_net)
        for _ in range(self.out_idx[hid_num]+1):
            self.out_net.append(nn.Sequential(Linear(hid_feats, out_feats), Sine(activate=False)))
        self.out_net = nn.Sequential(*self.out_net)

        with torch.no_grad():
            init_weight(self.in_net[0][0].weight, w0, True)                             ## 初始化 input
            init_bias(self.in_net[0][0].bias)
            for i, layer in enumerate(self.hid_net):                                    ## 初始化 hidden
                init_weight(layer[0].weight, w0, False)
                init_bias(layer[0].bias)
            for i, layer in enumerate(self.out_net):                                    ## 初始化 output
                init_weight(layer[0].weight, w0, False)
                init_bias(layer[0].bias)

    """ 前向传播 """
    def forward(self, input, c_depth, params=None, param_list=None):
        rec_feats = []
        final_output = 0
        output = self.in_net[0][0](input, params=self.get_subdict(params, f"in_net.0.0"))
        output = self.in_net[0][1](output)
        
        if plot_distribution: 
            res_list = param_list
        for i in range(self.out_idx[c_depth]+1):
            if i==0:
                for j, layer in enumerate(self.hid_net[:self.depths[0]]):
                    output = layer[0](output, params=self.get_subdict(params, f"hid_net.{j}.0"))
                    
                    if plot_distribution: 
                        if j in [c_depth-1]:
                            print(j, "$$$$$$$$$")
                            res_list.append(output)

                    output = layer[1](output)
            else:
                for k, layer in enumerate(self.hid_net[self.depths[i-1]:self.depths[i]]):
                    idx = self.depths[i-1] + k
                    output = layer[0](output, params=self.get_subdict(params, f"hid_net.{idx}.0"))
                    
                    if plot_distribution: 
                        if idx in [c_depth-1]:
                            print(idx, "&&&&&&&&&")
                            res_list.append(output)
                        
                    output = layer[1](output)
            
            output_feats = self.out_net[i][0](output, params=self.get_subdict(params, f"out_net.{i}.0"))
            output_feats = self.out_net[i][1](output_feats)
            
            if plot_distribution: 
                if i == self.out_idx[c_depth]:
                    res_list.append(output_feats)
                
            final_output = final_output + output_feats
            rec_feats.append(final_output)
        if plot_distribution: 
            for param in res_list:
                print(param.shape)
            plot_nums(res_list, bins=100, title=f"no_init_{c_depth}.jpg", y_label=f"stage {int(c_depth/2)}")
        
        return rec_feats

    """ 载入子网 """
    def load_net(self, net, s_width, s_depth):
        leftw, leftd, all = slice(None, s_width), slice(None, s_depth),slice(None)
        w_in, b_in, w_hid, b_hid, w_out, b_out = [], [], [], [], [], []
        with torch.no_grad():
            w_in.append(net.in_net[0][0].weight)                                    ## 获取 input
            b_in.append(net.in_net[0][0].bias)
            for i, layer in enumerate(net.hid_net[leftd]):                           ## 获取 hidden
                w_hid.append(layer[0].weight)
                b_hid.append(layer[0].bias)
            for i in range(self.out_idx[s_depth]+1):                                ## 获取 output
                w_out.append(net.out_net[i][0].weight)
                b_out.append(net.out_net[i][0].bias)
            self.in_net[0][0].weight[leftw, all] = w_in[0][leftw, all]                                      ## 载入 input[0]
            self.in_net[0][0].bias[leftw] = b_in[0][leftw]
            for i, layer in enumerate(self.hid_net[leftd]):                          ## 载入 hidden[0,1]
                layer[0].weight[leftw, leftw] = w_hid[i][leftw, leftw]
                layer[0].bias[leftw] = b_hid[i][leftw]
            for i in range(self.out_idx[s_depth]+1):
                self.out_net[i][0].weight[all, leftw] = w_out[i][all, leftw]
                self.out_net[i][0].bias[all] = b_out[i][all]

    """ 参数冻结 """
    def freeze_net(self, s_width, s_depth, freeze_out=True):
        leftd, leftw, all = slice(None, s_depth), slice(None, s_width), slice(None)                             
        self.in_net[0][0].weight.grad[leftw, all] = 0
        self.in_net[0][0].bias.grad[leftw] = 0
        for i, layer in enumerate(self.hid_net[leftd]):
            layer[0].weight.grad[leftw, all] = 0
            layer[0].bias.grad[leftw] = 0
        if freeze_out:
            for i in range(self.out_idx[s_depth]+1):
                self.out_net[i][0].weight.grad[all, leftw] = 0
                self.out_net[i][0].bias.grad[all] = 0

    """ 部分参数初始化 """
    def init_net(self, s_depth, init_hid=True, init_out=True, weight_scale=30*10, bias_scale=10):   ## s_dpeth=2 c_depth=4
        right, all = slice(s_depth, None), slice(None)                ## rightd[2,3]
        with torch.no_grad():
            if init_hid:
                for i, layer in enumerate(self.hid_net[right]):                                    ## hid_net[2,3]
                    init_weight(layer[0].weight, weight_scale)
                    init_bias(layer[0].bias, bias_scale)
            if init_out:
                init_weight(self.out_net[self.out_idx[s_depth]+1][0].weight, weight_scale)    ## self.out_idx[s_depth]+1][0]
                init_bias(self.out_net[self.out_idx[s_depth]+1][0].bias, bias_scale)  

    def prune_net(self, s_width, s_depth, prune_hidden=True, prune_output=True):
        leftd, all = slice(None, s_depth), slice(None)                      ## [0, 4]
        leftw, rightw = slice(None, s_width), slice(s_width, None)          ## [30]
        with torch.no_grad():
            for i, layer in enumerate(self.hid_net[leftd]):                 ## [0, 3]
                layer[0].weight[leftw, rightw] = 0
                if prune_hidden:
                    layer[0].weight[rightw, rightw] = 0
            if prune_output:                                                ## [0, 1]    [1, 2]
                for i in range(self.out_idx[s_depth]+1):
                    self.out_net[i][0].weight[all, rightw] = 0

""" 初始化权重 """
def init_weight(W, w0=30., is_first=False):
    fan_in = W.shape[1]
    u = 1 / fan_in if is_first else np.sqrt(6 / fan_in) / w0
    nn.init.uniform_(W, -u, u)

""" 初始化偏置 """
def init_bias(B, w0=1.):
    fan_in = B.shape[0]
    u = 1 / np.sqrt(fan_in) / w0
    nn.init.uniform_(B, -u, u)
    

## w ~ U(√(6/n), -√(6/n)) B = √(6/n)    

## n = 60   B = √(1/10) = 0.3   myB = 1 / 0.3 log(0.3 / )

## 1/B log(B / min(B, B)) = 0
## 1/B log()

## n = 60  B = 0.3   
## 60  -   12           log(60)√60 = 15     13.77
## 50  -   10           2√50 = 14           12
## 30  -                2√30 = 10

## √(6/n) / log(n)√(nh) = √(6)*√(/logn) / n
## x ~ [-1, 1]  w ~ [-B, B]  计算 wx+b 其中 w 和 b 越小，结果越小。
## 当宽度 n 越大，需要更大的值约束 B，使 wx+b稳定在一个范围。

#  B =                      while depth < min_depth

# 宽度深度越大