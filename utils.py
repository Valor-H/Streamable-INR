import os
import torch
import math
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataset import Dataset

""" 绘制元学习图像 """
def plot_img(imgs, psnrs, H, W, imgs_path, img_name):
    img_num = len(imgs)
    _, axes = plt.subplots(1, img_num, figsize=(20, 4))
    ax_titles = ['Init: ']
    for i in range(1, img_num-1):
        ax_titles.append(f"Step {i}: ")
    ax_titles.append('Ground True')
    for i, img in enumerate(imgs[:-1]):
        axes[i].set_axis_off()
        img = np.clip(img.view(H, W, 3).detach().cpu().numpy(), 0., 1.)
        axes[i].imshow(img)
        axes[i].set_title(ax_titles[i]+f"{psnrs[i]: .4f}", fontsize=20)
    imgs[-1] = np.clip(imgs[-1].view(H, W, 3).detach().cpu().numpy(), 0., 1.)
    axes[-1].set_axis_off()
    axes[-1].imshow(imgs[-1])
    axes[-1].set_title(ax_titles[-1], fontsize=20)
    plt.savefig(imgs_path + img_name)
    plt.show()
    plt.close()

""" 绘制分布图 """
def plot_nums(param_list, bins=80, title='demo.jpg', y_label="stage 1"):
    _, axes = plt.subplots(1, len(param_list), figsize=(20, 3))
    xlim = [1e-2, 4e-1, 6e-1]
    ylim = [2e1, 8e5, 6e4]
    for i, param in enumerate(param_list):
        print("here is i: ", i)
        axes[i].set_xlim([-xlim[i], xlim[i]])
        axes[i].tick_params(labelsize=12)
        axes[i].xaxis.set_ticks([-xlim[i], -xlim[i]/2, 0, xlim[i]/2, xlim[i]])
        axes[i].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        axes[i].hist(param.cpu().detach().flatten(), bins)
        if i == 0:
            axes[i].set_ylabel(y_label, fontsize=18)
        for label in axes[i].yaxis.get_ticklabels():
            label.set_fontsize(18)
        for label in axes[i].xaxis.get_ticklabels():
            label.set_fontsize(18)
    plt.savefig(f"./plot/plot_imgs/weight_imgs/{title}", dpi=300)
    
""" 写入字符串 """
def write_msg(msg, info_path, file_name):
    with open(info_path+file_name, "a", encoding="utf-8") as f:
        f.write(str(msg))
        f.close()

def get_res(args, idx, sub_idx, name):
    out_multi = 10                              ## 残差放大倍数
    bias = 0.05                                 ## 改变残差颜色
    img_type = ".png"  if args.mode in ["sd", "swd"] else ".jpg"
    mode_path = os.path.join(args.logs_path, f"{args.mode}")
    imgs_train_path = os.path.join(mode_path, "imgs/", "train/")
    c_width, c_depth = args.widths[idx], args.depths[idx]
    if idx>0:
        s_width, s_depth = args.widths[sub_idx], args.depths[sub_idx]
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(args.data_path, args.reshape, device)
    data_path = "./data/kodak/"
    img = dataset.get_by_path(data_path, f"{name}.png")
    feature = img["feature"].permute(1, 0).reshape(-1, img["H"], img["W"])
    save_path = f"./plot_imgs/res_img/{args.mode}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torchvision.utils.save_image(feature, os.path.join(save_path, f'{name}.jpg'))
    print("img shape", img["feature"].shape, img["name"])
    img_rec = dataset.get_by_path(imgs_train_path, f"{name}_w{c_width}d{c_depth}{img_type}")
    feature_hat = img_rec["feature"].permute(1, 0).reshape(-1, img_rec["H"], img_rec["W"])
    torchvision.utils.save_image(feature_hat, os.path.join(save_path, f"{name}_idx{idx}.jpg"))
    print("rec shape", img_rec["feature"].shape, img_rec["name"])
    feature_res = abs(feature - feature_hat) * out_multi
    torch.clamp(feature_res, -bias, 1-bias)
    feature_res[2] = feature_res[2] + bias
    torchvision.utils.save_image(feature_res, os.path.join(save_path, f"{name}_res_idx{idx}.jpg"))
    if idx>0:
        img_sub = dataset.get_by_path(imgs_train_path, f"{name}_w{s_width}d{s_depth}{img_type}")
        feature_sub = img_sub["feature"].permute(1, 0).reshape(-1, img_sub["H"], img_sub["W"])
        feature_inter = abs(feature_hat - feature_sub) * out_multi
        torch.clamp(feature_inter, -bias, 1-bias)
        feature_inter[2] = feature_inter[2] + bias
        torchvision.utils.save_image(feature_inter, os.path.join(save_path, f"{name}_inter_idx{sub_idx}_{idx}.jpg"))

if __name__=="__main__":
    a = [1, 2, 3]
    b = [0, 1, 2]
    ## 放于 train.py 中运行，主要借用一下传入的 args
    # for i in range(3):
    #     et_res(args, a[i], b[i], args.img_name)
    