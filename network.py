import os
import torch
import numpy as np
from torch import nn
from metrics import *
from utils import *
from collections import OrderedDict
from torchmeta.modules import MetaModule

""" 激活函数 """
class Sine(nn.Module):
    def __init__(self, w0=30., activate=True):
        super().__init__()
        self.w0 = w0
        self.activate = activate

    def forward(self, x):
        if self.activate:
            return torch.sin(self.w0*x)
        else:
            return x

""" 线性层 """
class Linear(nn.Linear, MetaModule):
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params["bias"]
        weight = params["weight"]
        output = input.matmul(weight.permute(-1, -2)) + bias
        return output

""" COIN, SW 网络 """
class SW(MetaModule):
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
    def __init__(self, args, in_feats, hid_feats, out_feats, hid_num, w0=30.):
        super().__init__()
        self.args = args
        self.state = args.state
        self.plot_distribution = args.plot_distribution
        self.depths = list(set(args.depths))
        (self.depths).sort()
        self.out_idx = {}
        for i, depth in enumerate(self.depths):         
            self.out_idx[depth] = i
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
            init_weight(self.in_net[0][0].weight, w0, True)
            init_bias(self.in_net[0][0].bias)
            for i, layer in enumerate(self.hid_net):
                init_weight(layer[0].weight, w0, False)
                init_bias(layer[0].bias)
            for i, layer in enumerate(self.out_net):
                init_weight(layer[0].weight, w0, False)
                init_bias(layer[0].bias)

    def forward(self, input, c_depth, params=None, param_list=None):
        rec_feats = []
        final_output = 0
        output = self.in_net[0][0](input, params=self.get_subdict(params, f"in_net.0.0"))
        output = self.in_net[0][1](output)
        
        if self.plot_distribution: 
            res_list = param_list
        for i in range(self.out_idx[c_depth]+1):
            if i==0:
                for j, layer in enumerate(self.hid_net[:self.depths[0]]):
                    output = layer[0](output, params=self.get_subdict(params, f"hid_net.{j}.0"))
                    
                    if self.plot_distribution: 
                        if j in [c_depth-1]:
                            print(j, "$$$$$$$$$")
                            res_list.append(output)

                    output = layer[1](output)
            else:
                for k, layer in enumerate(self.hid_net[self.depths[i-1]:self.depths[i]]):
                    idx = self.depths[i-1] + k
                    output = layer[0](output, params=self.get_subdict(params, f"hid_net.{idx}.0"))
                    
                    if self.plot_distribution: 
                        if idx in [c_depth-1]:
                            print(idx, "&&&&&&&&&")
                            res_list.append(output)
                        
                    output = layer[1](output)
            
            output_feats = self.out_net[i][0](output, params=self.get_subdict(params, f"out_net.{i}.0"))
            output_feats = self.out_net[i][1](output_feats)
            
            if self.plot_distribution: 
                if i == self.out_idx[c_depth]:
                    res_list.append(output_feats)
                
            final_output = final_output + output_feats
            rec_feats.append(final_output)
        if self.plot_distribution: 
            for param in res_list:
                print(param.shape)
            plot_nums(res_list, bins=100, title=f"no_init_{c_depth}.jpg", y_label=f"stage {int(c_depth/2)}")
        
        return rec_feats

    def load_net(self, net, s_width, s_depth):
        leftw, leftd, all = slice(None, s_width), slice(None, s_depth),slice(None)
        w_in, b_in, w_hid, b_hid, w_out, b_out = [], [], [], [], [], []
        with torch.no_grad():
            w_in.append(net.in_net[0][0].weight)
            b_in.append(net.in_net[0][0].bias)
            for i, layer in enumerate(net.hid_net[leftd]):
                w_hid.append(layer[0].weight)
                b_hid.append(layer[0].bias)
            for i in range(self.out_idx[s_depth]+1):
                w_out.append(net.out_net[i][0].weight)
                b_out.append(net.out_net[i][0].bias)
            self.in_net[0][0].weight[leftw, all] = w_in[0][leftw, all]
            self.in_net[0][0].bias[leftw] = b_in[0][leftw]
            for i, layer in enumerate(self.hid_net[leftd]):
                layer[0].weight[leftw, leftw] = w_hid[i][leftw, leftw]
                layer[0].bias[leftw] = b_hid[i][leftw]
            for i in range(self.out_idx[s_depth]+1):
                self.out_net[i][0].weight[all, leftw] = w_out[i][all, leftw]
                self.out_net[i][0].bias[all] = b_out[i][all]

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

    def init_net(self, s_depth, init_hid=True, init_out=True, weight_scale=30*10, bias_scale=10):
        right, all = slice(s_depth, None), slice(None)  
        with torch.no_grad():
            if init_hid:
                for i, layer in enumerate(self.hid_net[right]):  
                    init_weight(layer[0].weight, weight_scale)
                    init_bias(layer[0].bias, bias_scale)
            if init_out:
                init_weight(self.out_net[self.out_idx[s_depth]+1][0].weight, weight_scale)
                init_bias(self.out_net[self.out_idx[s_depth]+1][0].bias, bias_scale)  

    def prune_net(self, s_width, s_depth, prune_hidden=True, prune_output=True):
        leftd, all = slice(None, s_depth), slice(None)                 
        leftw, rightw = slice(None, s_width), slice(s_width, None)        
        with torch.no_grad():
            for i, layer in enumerate(self.hid_net[leftd]):        
                layer[0].weight[leftw, rightw] = 0
                if prune_hidden:
                    layer[0].weight[rightw, rightw] = 0
            if prune_output:                       
                for i in range(self.out_idx[s_depth]+1):
                    self.out_net[i][0].weight[all, rightw] = 0

def init_weight(W, w0=30., is_first=False):
    fan_in = W.shape[1]
    u = 1 / fan_in if is_first else np.sqrt(6 / fan_in) / w0
    nn.init.uniform_(W, -u, u)

def init_bias(B, w0=1.0):
    fan_in = B.shape[0]
    u = 1 / np.sqrt(fan_in) / w0
    nn.init.uniform_(B, -u, u)
    