import os
import gc
import copy
import torch
import time
from tqdm.autonotebook import tqdm
from dataset import Dataset
from network import SW, SWD
from maml import MAML 
from metrics import *
from utils import *
from quant_entropy import Quentizer

class ImageTrainer():
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.state = args.state
        self.widths = args.widths
        self.depths = args.depths
        self.eval_all = args.eval_all
        self.logs_inter = args.logs_inter
        self.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
        self.dataset = Dataset(args.data_path, args.reshape, self.device)
        self.meta_dataset = Dataset(args.meta_path, args.reshape, self.device)
        self.img_num = len(self.meta_dataset) if self.state=="meta" else len(self.dataset)
        self.mode_path = os.path.join(self.args.logs_path, f"{args.mode}")
        self.eval_path = os.path.join(self.mode_path, "eval/")
        self.imgs_train_path = os.path.join(self.mode_path, "imgs/", "train/")
        self.mods_train_path = os.path.join(self.mode_path, "mods/", "train/")
        self.imgs_mtrain_path = os.path.join(self.mode_path, "imgs/", "mtrain/")
        self.mods_mtrain_path = os.path.join(self.mode_path, "mods/", "mtrain/")
        self.mods_meta_path = os.path.join(self.mode_path, "mods/", "meta/")
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)
        if not os.path.exists(self.imgs_train_path):
            os.makedirs(self.imgs_train_path)
        if not os.path.exists(self.imgs_mtrain_path):
            os.makedirs(self.imgs_mtrain_path)
        if not os.path.exists(self.mods_train_path):
            os.makedirs(self.mods_train_path)
        if not os.path.exists(self.mods_mtrain_path):
            os.makedirs(self.mods_mtrain_path)

    def coin_train(self, idx):
        c_width, c_depth = self.widths[idx], self.depths[idx]                               ##! 当前宽度，当前深度
        cur_net = SW(self.args, 2, c_width, 3, c_depth).to(self.device)                     ##! 当前模型
        if self.state in ["train", "mtrain"]:                                               ##! 训练模式
            all_psnrs, all_ssims, all_lpips = [], [], []
            for step, img_data in enumerate(self.dataset):
                name = img_data["name"]
                if self.args.img_name != None:
                    if name not in self.args.img_name:
                        continue
                print(f"[mode:{self.mode}\tstate:{self.state}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}\t]")
                write_msg(f"[mode:{self.mode}\tstate:{self.state}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nPSNR=[", self.eval_path, f"{self.state}_psnr.txt")
                if self.eval_all:
                    write_msg(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nSSIM=[", self.eval_path, f"{self.state}_ssim.txt")
                    write_msg(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nLPIPS=[", self.eval_path, f"{self.state}_lpips.txt")
                train_net = copy.deepcopy(cur_net)
                if self.state == "mtrain":
                    train_net.load_state_dict(torch.load(os.path.join(self.mods_meta_path, f"meta_w{c_width}d{c_depth}.pth")))
                each_psnr, each_ssim, each_lpips = self.training(train_net, img_data, c_width, c_depth)
                all_psnrs.append(each_psnr)
                if self.eval_all:
                    all_ssims.append(each_ssim)
                    all_lpips.append(each_lpips)
            write_msg(f"\n", self.eval_path, f"{self.state}_psnr.txt")
            metrics(all_psnrs, "psnr", self.mode, c_width, c_depth, self.eval_path, self.state)
            if self.eval_all:
                write_msg(f"\n", self.eval_path, f"{self.state}_ssim.txt")
                write_msg(f"\n", self.eval_path, f"{self.state}_lpips.txt")
                metrics(all_ssims, "ssim", self.mode, c_width, c_depth, self.eval_path, self.state)
                metrics(all_lpips, "lpips", self.mode, c_width, c_depth, self.eval_path, self.state)
        if self.state in ["meta"]:                                                          ##! 元学习模式
            cur_net = copy.deepcopy(cur_net)
            meta_net = MAML(self.args, cur_net, is_multi=False, device=self.device)
            optimize = torch.optim.Adam(params=meta_net.parameters(), lr=self.args.out_lr)
            with tqdm(total=self.args.out_epochs*self.img_num) as pbar:
                for out_epoch in range(self.args.out_epochs):
                    for step, img_data in enumerate(self.meta_dataset):
                        loss = meta_net(img_data, step+out_epoch*self.img_num)
                        optimize.zero_grad()
                        loss.backward()
                        optimize.step()
                        if step and not step % self.logs_inter:
                            torch.save(meta_net.net.state_dict(), os.path.join(meta_net.mods_meta_path, f"meta_w{c_width}d{c_depth}.pth"))
                            gc.collect()
                            torch.cuda.empty_cache()
                        pbar.update(1)                
        if self.state in ["quant"]:
            self.quant_entropy(c_width, c_depth, cur_net)

    def sw_train(self, idx):
        c_width, c_depth = self.widths[idx], self.depths[idx]
        s_width, s_depth = (self.widths[idx-1], self.depths[idx-1]) if idx>0 else (0, 0)
        cur_net = SW(self.args, 2, c_width, 3, c_depth).to(self.device)
        if idx>0:
            sub_net = SW(self.args, 2, s_width, 3, c_depth).to(self.device)
        if self.state in ["train", "mtrain"]:
            all_psnrs, all_ssims, all_lpips = [], [], []
            for step, img_data in enumerate(self.dataset):
                name = img_data["name"]
                if self.args.img_name != None:
                    if name not in self.args.img_name:
                        continue
                print(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}\t]")
                write_msg(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nPSNR=[", self.eval_path, f"{self.state}_psnr.txt")
                if self.eval_all:
                    write_msg(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nSSIM=[", self.eval_path, f"{self.state}_ssim.txt")
                    write_msg(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nLPIPS=[", self.eval_path, f"{self.state}_lpips.txt")
                train_net = copy.deepcopy(cur_net)
                if self.state == "train":
                    if idx>0:
                        sub_net.load_state_dict(torch.load(os.path.join(self.mods_train_path, f"{name}_w{s_width}d{c_depth}.pth")))
                        train_net.load_net(sub_net, s_width)
                        train_net.prune_net(s_width, True, True)
                elif self.state == "mtrain":
                    if idx==0:
                        train_net.load_state_dict(torch.load(os.path.join(self.mods_meta_path, f"meta_w{c_width}d{c_depth}.pth")))
                    elif idx>0:
                        sub_net.load_state_dict(torch.load(os.path.join(self.mods_mtrain_path, f"{name}_w{s_width}d{s_depth}.pth")))
                        train_net.load_net(sub_net, s_width)
                        train_net.prune_net(s_width, True, True)
                each_psnr, each_ssim, each_lpips = self.training(train_net, img_data, c_width, c_depth, s_width, s_depth, True)
                all_psnrs.append(each_psnr)
                if self.eval_all:
                    all_ssims.append(each_ssim)
                    all_lpips.append(each_lpips)
            write_msg(f"\n", self.eval_path, f"{self.state}_psnr.txt")
            metrics(all_psnrs, "psnr", self.mode, c_width, c_depth, self.eval_path, self.state)
            if self.eval_all:
                write_msg(f"\n", self.eval_path, f"{self.state}_ssim.txt")
                write_msg(f"\n", self.eval_path, f"{self.state}_lpips.txt")
                metrics(all_ssims, "ssim", self.mode, c_width, c_depth, self.eval_path, self.state)
                metrics(all_lpips, "lpips", self.mode, c_width, c_depth, self.eval_path, self.state)
        if self.state in ["meta"] and idx==0:
            cur_net = copy.deepcopy(cur_net)
            meta_net = MAML(self.args, cur_net, is_multi=False, device=self.device)
            optimize = torch.optim.Adam(params=meta_net.parameters(), lr=self.args.out_lr)
            with tqdm(total=self.args.out_epochs*self.img_num) as pbar:
                for out_epoch in range(self.args.out_epochs):
                    for step, img_data in enumerate(self.meta_dataset):
                        loss = meta_net(img_data, step+out_epoch*self.img_num)
                        optimize.zero_grad()
                        loss.backward()
                        optimize.step()
                        if step and not step % self.logs_inter:
                            torch.save(meta_net.net.state_dict(), os.path.join(meta_net.mods_meta_path, f"meta_w{c_width}d{c_depth}.pth"))
                            gc.collect()
                            torch.cuda.empty_cache()
                        pbar.update(1)
                        if step == len(self.meta_dataset) - 1:
                            print("save last mod")
                            torch.save(meta_net.net.state_dict(), os.path.join(meta_net.mods_meta_path, f"meta_w{c_width}d{c_depth}.pth"))
        if self.state in ["quant_entropy"]:
            self.quant_entropy(c_width, c_depth, cur_net)

    def sd_train(self, idx):
        c_width, c_depth = (self.widths[idx], self.depths[idx])
        s_width, s_depth = (self.widths[idx-1], self.depths[idx-1]) if idx>0 else (0, 0)
        cur_net = SWD(self.args, 2, c_width, 3, c_depth).to(self.device)
        st_m, st_c = len(set(self.depths)), idx + 1
        sc = math.sqrt((st_c + st_m) / (st_m * c_width))
        if idx>0:
            sub_net = SWD(self.args, 2, c_width, 3, s_depth).to(self.device)
        if self.state in ["train", "mtrain"]:
            all_psnrs, all_ssims, all_lpips = [], [], []
            for step, img_data in enumerate(self.dataset):
                name = img_data["name"]
                if self.args.img_name != None:
                    if name not in self.args.img_name:
                        continue
                print(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}\t]")
                write_msg(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nPSNR=[", self.eval_path, f"{self.state}_psnr.txt")
                if self.eval_all:
                    write_msg(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nSSIM=[", self.eval_path, f"{self.state}_ssim.txt")
                    write_msg(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nLPIPS=[", self.eval_path, f"{self.state}_lpips.txt")
                train_net = copy.deepcopy(cur_net)
                if self.state == "train":
                    if idx>0:
                        sub_net.load_state_dict(torch.load(os.path.join(self.mods_train_path, f"{name}_w{c_width}d{s_depth}.pth")))
                        train_net.load_net(sub_net, s_width, s_depth)
                        train_net.init_net(s_depth, True, True, 30.*sc, sc)
                elif self.state == "mtrain":
                    if idx==0:
                        train_net.load_state_dict(torch.load(os.path.join(self.mods_meta_path, f"meta_w{c_width}d{c_depth}.pth")))
                    elif idx>0:
                        sub_net.load_state_dict(torch.load(os.path.join(self.mods_mtrain_path, f"{name}_w{c_width}d{s_depth}.pth")))
                        train_net.load_net(sub_net, s_width, s_depth)
                        train_net.init_net(s_depth, True, True, 30.*sc[idx], sc[idx])
                each_psnr, each_ssim, each_lpips = self.training(train_net, img_data, c_width, c_depth, s_width, s_depth, True)
                all_psnrs.append(each_psnr)
                if self.eval_all:
                    all_ssims.append(each_ssim)
                    all_lpips.append(each_lpips)
            write_msg(f"\n", self.eval_path, f"{self.state}_psnr.txt")
            metrics(all_psnrs, "psnr", self.mode, c_width, c_depth, self.eval_path, self.state)
            if self.eval_all:
                write_msg(f"\n", self.eval_path, f"{self.state}_ssim.txt")
                write_msg(f"\n", self.eval_path, f"{self.state}_lpips.txt")
                metrics(all_ssims, "ssim", self.mode, c_width, c_depth, self.eval_path, self.state)
                metrics(all_lpips, "lpips", self.mode, c_width, c_depth, self.eval_path, self.state)
        if self.state in ["meta"] and idx==0:
            cur_net = copy.deepcopy(cur_net)
            meta_net = MAML(self.args, cur_net, is_multi=True, device=self.device)
            optimize = torch.optim.Adam(params=meta_net.parameters(), lr=self.args.out_lr)
            with tqdm(total=self.args.out_epochs*self.img_num) as pbar:
                for out_epoch in range(self.args.out_epochs):
                    for step, img_data in enumerate(self.meta_dataset):
                        if step > 100000:
                            break
                        loss = meta_net(img_data, step+out_epoch*self.img_num)
                        optimize.zero_grad()
                        loss.backward()
                        optimize.step()
                        if step and (not step % self.logs_inter):
                            torch.save(meta_net.net.state_dict(), os.path.join(meta_net.mods_meta_path, f"meta_w{c_width}d{c_depth}.pth"))
                            gc.collect()
                            torch.cuda.empty_cache()
                        pbar.update(1)
        if self.state in ["quant_entropy"]:
            self.quant_entropy(c_width, c_depth, cur_net)
        if self.args.plot_distribution:
            self.plot_distritubion(idx, c_width, c_depth, s_width, s_depth, cur_net)

    def swd_train(self, idx):
        c_width, c_depth = self.widths[idx], self.depths[idx]
        s_width, s_depth = (self.widths[idx-1], self.depths[idx-1]) if idx>0 else (0, 0)
        cur_net = SWD(self.args, 2, c_width, 3, c_depth).to(self.device)
        st_m, st_c = len(set(self.depths)), idx + 1
        sc = math.sqrt((st_c + st_m) / (st_m * c_width))
        if idx>0:
            sub_net = SWD(self.args, 2, s_width, 3, s_depth).to(self.device)
        if self.state in ["train", "mtrain"]:
            all_psnrs, all_ssims, all_lpips = [], [], []
            for step, img_data in enumerate(self.dataset):
                name = img_data["name"]
                if self.args.img_name != None:
                    if name not in self.args.img_name:
                        continue
                print(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}\t{s_width}\t{s_depth}]")
                write_msg(f"[mode:{self.mode}\tname:{name}\tc_width:{c_width}\tc_depth:{c_depth}]:\nPSNR=[", self.eval_path, f"{self.state}_psnr.txt")
                train_net = copy.deepcopy(cur_net)
                if self.state == "train":
                    if idx>0:
                        sub_net.load_state_dict(torch.load(os.path.join(self.mods_train_path, f"{name}_w{s_width}d{s_depth}.pth")))
                        train_net.load_net(sub_net, s_width, s_depth)
                        if c_depth - s_depth > 0:
                            train_net.init_net(s_depth, True, True, 30.*sc, sc)
                        if c_width - s_width > 0:
                            train_net.prune_net(s_width, s_depth, True, True)
                elif self.state == "mtrain":
                    if idx==0:
                        train_net.load_state_dict(torch.load(os.path.join(self.mods_meta_path, f"meta_w{c_width}d{c_depth}.pth")))
                    elif idx>0:
                        sub_net.load_state_dict(torch.load(os.path.join(self.mods_mtrain_path, f"{name}_w{s_width}d{s_depth}.pth")))
                        train_net.load_net(sub_net, s_width, s_depth)
                        train_net.init_net(s_depth, True, True, 30.*sc[idx], sc[idx])
                    if (c_depth-s_depth)>0 and idx>0:
                        train_net.prune_net(s_width, s_depth, True, True)
                each_psnr, each_ssim, each_lpips = self.training(train_net, img_data, c_width, c_depth, s_width, s_depth, True)
                all_psnrs.append(each_psnr)
                if self.eval_all:
                    all_ssims.append(each_ssim)
                    all_lpips.append(each_lpips)
            write_msg(f"\n", self.eval_path, f"{self.state}_psnr.txt")
            metrics(all_psnrs, "psnr", self.mode, c_width, c_depth, self.eval_path, self.state)
            if self.eval_all:
                write_msg(f"\n", self.eval_path, f"{self.state}_ssim.txt")
                write_msg(f"\n", self.eval_path, f"{self.state}_lpips.txt")
                metrics(all_ssims, "ssim", self.mode, c_width, c_depth, self.eval_path, self.state)
                metrics(all_lpips, "lpips", self.mode, c_width, c_depth, self.eval_path, self.state)   
        if self.state in ["meta"] and idx==0:
            cur_net = copy.deepcopy(cur_net)
            cur_net.load_state_dict(torch.load(os.path.join("logs/Meta-Learning/div2k/sw/mods/meta", f"{name}_w{c_width}d{c_depth}.pth")))
            meta_net = MAML(self.args, cur_net, is_multi=True, device=self.device)
            optimize = torch.optim.Adam(params=meta_net.parameters(), lr=self.args.out_lr)
            with tqdm(total=self.args.out_epochs*self.img_num) as pbar:
                for out_epoch in range(self.args.out_epochs):
                    for step, img_data in enumerate(self.meta_dataset):
                        loss = meta_net(img_data, step+out_epoch*self.img_num)
                        optimize.zero_grad()
                        loss.backward()
                        optimize.step()
                        if step and (not step % self.logs_inter):
                            torch.save(meta_net.net.state_dict(), os.path.join(meta_net.mods_meta_path, f"meta_w{c_width}d{c_depth}.pth"))
                            gc.collect()
                            torch.cuda.empty_cache()
                        pbar.update(1)                
        if self.state in ["quant_entropy"]:
            self.quant_entropy(c_width, c_depth, cur_net)
        if self.args.plot_distribution:
            self.plot_distritubion(idx, c_width, c_depth, s_width, s_depth, cur_net)
    
    def training(self, train_net, img_data, c_width, c_depth, s_width=0, s_depth=0, last_loss=True):
        train_net.train()
        optimize = torch.optim.Adam(train_net.parameters(), self.args.lr, amsgrad=True)
        coord, feature, H, W = img_data["coord"], img_data["feature"], img_data["H"], img_data["W"]
        each_psnr, each_ssim, each_lpips = [], [], []
        best_psnr, best_ssim, best_lpips = torch.zeros(3).to(self.device)
        with tqdm(total=self.args.epochs) as pbar:
            for epoch in range(1, self.args.epochs+1):
                if self.args.mode in ["coin", "sw"]:
                    feature_hat = train_net(coord)
                    loss = cal_loss(feature, feature_hat)
                    optimize.zero_grad()
                    loss.backward()
                    if s_width>0:
                        train_net.freeze_net(s_width)
                    optimize.step()
                    if epoch and (not epoch % self.logs_inter):
                        cur_psnr = cal_psnr(loss)
                        best_psnr = max(cur_psnr, best_psnr)
                        if self.eval_all:
                            best_ssim = cal_ssim(feature, feature_hat, H, W)
                            best_lpips = cal_lpips(feature, feature_hat, H, W)
                        eval_width, eval_depth = c_width, c_depth
                        if epoch > self.args.epochs * 0.1:
                            if (best_psnr == cur_psnr):
                                if self.state == "train":
                                    eval_imgs_path, eval_mods_path = self.imgs_train_path, self.mods_train_path
                                elif self.state == "mtrain":
                                    eval_imgs_path, eval_mods_path = self.imgs_mtrain_path, self.mods_mtrain_path
                                eval_net(train_net, img_data, eval_width, eval_depth, eval_imgs_path, eval_mods_path)
                        each_psnr.append(best_psnr.item())
                        if self.eval_all:
                            each_ssim.append(best_ssim.item())
                            each_lpips.append(best_lpips.item())
                        if self.eval_all:
                            print(f"Epoch[{epoch}/{self.args.epochs}]\tPSNR:{best_psnr}\tSSIM:{best_ssim}\tLPIPS:{best_lpips.item()}")
                        else:
                            print(f"Epoch[{epoch}/{self.args.epochs}]\tB_PSNR:{best_psnr}\tC_PSNR:{cur_psnr}")
                        write_msg(f"{best_psnr: .4f},", self.eval_path, f"{self.state}_psnr.txt")
                        if self.eval_all:
                            write_msg(f"{best_ssim: .4f},", self.eval_path, f"{self.state}_ssim.txt")
                            write_msg(f"{best_lpips.item(): .4f},", self.eval_path, f"{self.state}_lpips.txt")
                        gc.collect()
                        torch.cuda.empty_cache()
                    pbar.update(1)
                elif self.args.mode in ["sd", "swd"]:
                    loss = []
                    feature_hat = train_net(coord, c_depth)
                    for feat in feature_hat:
                        loss.append(cal_loss(feature, feat))
                    if last_loss:
                        all_loss = loss[-1]
                    else:
                        all_loss = sum(loss)/len(loss)
                    optimize.zero_grad()
                    all_loss.backward()
                    if s_width>0 and (c_width-s_width)>0:
                        train_net.freeze_net(s_width, s_depth, True)
                    optimize.step()
                    if epoch and (not epoch % self.logs_inter):
                        cur_psnr = cal_psnr(loss[-1])
                        best_psnr = max(cur_psnr, best_psnr)
                        if self.eval_all:
                            best_ssim = cal_ssim(feature, feature_hat[-1], H, W)
                            best_lpips = cal_lpips(feature, feature_hat[-1], H, W)
                        if epoch > self.args.epochs * 0.1:
                            if (best_psnr == cur_psnr):
                                if self.state == "train":
                                    eval_imgs_path, eval_mods_path = self.imgs_train_path, self.mods_train_path
                                elif self.state == "mtrain":
                                    eval_imgs_path, eval_mods_path = self.imgs_mtrain_path, self.mods_mtrain_path
                                eval_net_multi(train_net, img_data, c_width, c_depth, eval_imgs_path, eval_mods_path)
                        each_psnr.append(best_psnr.item())
                        if self.eval_all:
                            each_ssim.append(best_ssim.item())
                            each_lpips.append(best_lpips.item())
                        if self.eval_all:
                            print(f"Epoch[{epoch}/{self.args.epochs}]\tPSNR:{best_psnr}\tSSIM:{best_ssim}\tLPIPS:{best_lpips.item()}")
                        else:
                            print(f"Epoch[{epoch}/{self.args.epochs}]\tB_PSNR:{best_psnr}\tC_PSNR:{cur_psnr}")                    
                        write_msg(f"{best_psnr: .4f},", self.eval_path, f"{self.state}_psnr.txt")
                        if self.eval_all:
                            write_msg(f"{best_ssim: .4f},", self.eval_path, f"{self.state}_ssim.txt")
                            write_msg(f"{best_lpips.item(): .4f},", self.eval_path, f"{self.state}_lpips.txt")
                        gc.collect()
                        torch.cuda.empty_cache()
                    pbar.update(1)
            write_msg(f"]\n", self.eval_path, f"{self.state}_psnr.txt")
            if self.eval_all:
                write_msg(f"]\n", self.eval_path, f"{self.state}_ssim.txt")
                write_msg(f"]\n", self.eval_path, f"{self.state}_lpips.txt")
        return each_psnr, each_ssim, each_lpips
    
    def quant_entropy(self, c_width, c_depth, cur_net):
        quantizer = Quentizer(self.args)
        ori_psnr_list, quant_psnr_list, down_psnr_list = [], [], []
        entropy_list = []
        for num_bit in self.args.num_bits:
            print("num_bit:", num_bit)
            ori_psnr_bit, quant_psnr_bit, down_psnr_bit = [], [], []
            entropy_bit = []
            for step, img_data in enumerate(self.dataset):
                coord, feature, name = img_data["coord"], img_data["feature"], img_data["name"]
                quant_net = copy.deepcopy(cur_net)
                quant_net.load_state_dict(torch.load(os.path.join(self.mods_train_path, f"{name}_w{c_width}d{c_depth}.pth")))
                if self.args.mode in ["coin", "sw"]:
                    ori_psnr = cal_psnr(cal_loss(feature, quant_net(coord)))
                elif self.args.mode in ["sd", "swd"]:
                    ori_psnr = cal_psnr(cal_loss(feature, (quant_net(coord, c_depth)[-1])))
                ori_psnr_bit.append(ori_psnr.item())
                quantizer.get_mean_and_std(quant_net)
                dequatized_net, num_bit_entropy = quantizer.quantize(quant_net, num_bit)
                if self.args.mode in ["coin", "sw"]:
                    quant_psnr = cal_psnr(cal_loss(feature, dequatized_net(coord)))
                elif self.args.mode in ["sd", "swd"]:
                    quant_psnr = cal_psnr(cal_loss(feature, (dequatized_net(coord, c_depth)[-1])))
                quant_psnr_bit.append(quant_psnr.item())
                down_psnr_bit.append((quant_psnr-ori_psnr).item())
                entropy_bit.append(num_bit_entropy)
            ori_psnr_list.append(np.mean(ori_psnr_bit))
            quant_psnr_list.append(np.mean(quant_psnr_bit))
            down_psnr_list.append(np.mean(down_psnr_bit))
            entropy_list.append(np.mean(entropy_bit))
        print("ori:[", end="")
        for psnr in ori_psnr_list:
            print(f"{psnr:.4f}", end=", ")
        print("]\nquant:[", end="")
        for psnr in quant_psnr_list:
            print(f"{psnr:.4f}", end=", ")
        print("]\ndown:[", end="")
        for down in down_psnr_list:
            print(f"{down:.4f}", end=", ")
        print("]\nquant:[", end="")
        for num in entropy_list:
            print(f"{num:.2f}", end=", ")
        print("]\n")

    def plot_distritubion(self, idx, c_width, c_depth, s_width, s_depth, cur_net):
        for _, img_data in enumerate(self.dataset):
            name = img_data["name"]
            if self.args.img_name != None:
                if name not in self.args.img_name:
                    continue
            coord, feature, name = img_data["coord"], img_data["feature"], img_data["name"]
            start_time = time.time()
            net = copy.deepcopy(cur_net)
            st_m, st_c = len(set(self.depths)), idx + 1
            sc = math.sqrt((st_c + st_m) / (st_m * c_width))
            if idx > 0:
                net.init_net(s_depth, True, True, 30.*sc, sc)
            param_list = []
            for i, (namee, param) in enumerate(net.named_parameters()):
                print(i, " ", namee)
                if i in [2 * c_depth]:
                    param_list.append(param)
                    print(i, "###############")
            mod_path = "/home/xch/codes/meta-swd/logs/logs_tmp/sd/mods/train"
            cur_net.load_state_dict(torch.load(os.path.join(self.mods_train_path, f"{name}_w{c_width}d{c_depth}.pth")))
            cur_net(coord, c_depth, param_list=param_list)
            print("花费时间", time.time() - start_time)