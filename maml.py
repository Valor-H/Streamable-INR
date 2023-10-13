import os
import torch
from torch import nn
from eval import cal_loss, cal_psnr, plot_img
from torchmeta.modules import MetaModule
from collections import OrderedDict

class MAML(MetaModule):
    def __init__(self, args, net, is_multi=False, device=None):
        super().__init__()
        self.args = args
        self.net = net
        self.is_multi = is_multi
        self.save_idx = 1
        self.device = device
        if args.lr_type=="static":
            self.register_buffer('in_lr', torch.Tensor([args.in_lr]))
        elif args.lr_type=="param":
            self.in_lr = nn.ParameterList([])
            for param in net.parameters():
                self.in_lr.append(nn.Parameter(torch.ones(param.size())*args.in_lr))
        elif args.lr_type == "step_param":
            self.in_lr = nn.ModuleList([])
            for name, param in net.meta_named_parameters():
                self.in_lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size())*args.in_lr)
                    for _ in range(args.in_epochs)]))
        self.in_lr = self.in_lr.to(self.device)
        self.mode_path = os.path.join(args.logs_path, f"{args.mode}")
        self.imgs_meta_path = os.path.join(self.mode_path, "imgs/", "meta/")
        self.mods_meta_path = os.path.join(self.mode_path, "mods/", "meta/")
        if not os.path.exists(self.imgs_meta_path):
            os.makedirs(self.imgs_meta_path)
        if not os.path.exists(self.mods_meta_path):
            os.makedirs(self.mods_meta_path)

    def forward(self, img_data, step):
        coord, feature, name, H, W = img_data["coord"], img_data["feature"], img_data["name"], img_data["H"], img_data["W"]
        fast_params, all_imgs, all_psnrs = self.in_loop(img_data, step)
        if self.is_multi:
            final_feature = self.net(coord, self.net.c_depth, params=fast_params)[-1]
        else:
            final_feature = self.net(coord, params=fast_params)
        final_loss = cal_loss(feature, final_feature)
        final_psnr = cal_psnr(final_loss)
        if (not step % self.args.logs_inter or step == self.save_idx):
            all_imgs.append(final_feature)
            all_imgs.append(feature)
            all_psnrs.append(final_psnr.item())
            plot_path, img_name = self.imgs_meta_path, f"meta_w{self.net.c_width}d{self.net.c_depth}.jpg"
            plot_img(all_imgs, all_psnrs, H, W, plot_path, img_name)
            print(f"step:{step}\tall_psnrs: [", end="")
            for num in all_psnrs:
                print(f"{num: .4f},\t", end='')
            print("]")
        return final_loss
    
    def in_loop(self, img_data, step):
        coord, feature = img_data["coord"], img_data["feature"]
        with torch.enable_grad():
            fast_params = OrderedDict()
            for name, param in self.net.meta_named_parameters():
                fast_params[name] = param
            all_imgs, all_psnrs = [], []
            for in_step in range(self.args.in_epochs):
                if self.is_multi:
                    each_feature = self.net(coord, self.net.c_depth, params=fast_params)[-1]
                else:
                    each_feature = self.net(coord, params=fast_params)
                each_loss = cal_loss(feature, each_feature)
                each_psnr = cal_psnr(each_loss)
                if (not step % self.args.logs_inter or step == self.save_idx):
                    all_imgs.append(each_feature)
                    all_psnrs.append(each_psnr.item())
                fast_params = self.in_loop_step(each_loss, fast_params, in_step)
        return fast_params, all_imgs, all_psnrs
    
    def in_loop_step(self, loss, fast_params, in_step):
        grads = torch.autograd.grad(loss, fast_params.values(), create_graph=True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(fast_params.items(), grads)):
            if self.args.lr_type=="static":
                params[name] = param - self.in_lr * grad
            elif self.args.lr_type=="param":
                params[name] = param - self.in_lr[i] * grad
            elif self.args.lr_type=="step_param":
                lr = self.in_lr[i][in_step]
                params[name] = param - lr * grad
        return params
    