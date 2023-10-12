import os
import torch
import math
import torchvision
import numpy as np
import lpips
import matplotlib.pyplot as plt
import torch.nn.functional as F

""" 评估网络 """
def eval_net(train_net, img_data, c_width, c_depth, imgs_path, mods_path):
    coord, feature, name, H, W = img_data["coord"], img_data["feature"], img_data["name"], img_data["H"], img_data["W"]
    feature_hat = train_net(coord).permute(1, 0).reshape(-1, H, W)
    torchvision.utils.save_image(feature_hat, os.path.join(imgs_path, f'{name}_w{c_width}d{c_depth}.jpg'))
    torch.save(train_net.state_dict(), os.path.join(mods_path, f"{name}_w{c_width}d{c_depth}.pth"))

def eval_net_multi(train_net, img_data, c_width, c_depth, imgs_path, mods_path):
    coord, feature, name, H, W = img_data["coord"], img_data["feature"], img_data["name"], img_data["H"], img_data["W"]
    feature_hat = (train_net(coord, c_depth)[-1]).permute(1, 0).reshape(-1, H, W)
    torchvision.utils.save_image(feature_hat, os.path.join(imgs_path, f'{name}_w{c_width}d{c_depth}.png'))
    torch.save(train_net.state_dict(), os.path.join(mods_path, f"{name}_w{c_width}d{c_depth}.pth"))

## 根据给定的数据，求其均值，写入到文件中
def metrics(metrics_list, title, mode, c_width, c_depth, eval_path, state):
    file_name = state+"_"+title+"_"+"mean"
    metrics_mean = cal_mean(metrics_list, title)
    write_msg(f"[---MEAN--- mode:{mode}\tc_width:{c_width}\tc_depth:{c_depth}]:\nPSNR=[", eval_path, f"{file_name}.txt")
    for mean in metrics_mean:
        write_msg(f"{mean: .4f},", eval_path, f"{file_name}.txt")
    write_msg(f"]\n\n", eval_path, f"{file_name}.txt")

def cal_mean(all_list, title, drop_list=[]):
    sum_list = 0
    for i, list in enumerate(all_list):
        if (i+1) not in drop_list:
            sum_list += np.array(list)
    len_num = len(all_list) - len(drop_list)
    mean_list = sum_list / len_num
    print(f"{title}: len_num: {len_num} [", end='')
    [print(f"{num: .4f},", end='') for num in mean_list]
    print("]")
    return mean_list

def cal_loss(feature, feature_hat):
    loss = torch.mean((feature-feature_hat)**2)
    return loss

def cal_psnr(loss):
    psnr = 10 * torch.log10(1 / loss)
    return psnr

def cal_ssim(feature, feature_hat, H, W, w_size=11, size_average=True, full=False):
    feature = feature.permute(1, 0).reshape(1, -1, H, W).detach().cpu()
    feature_hat = feature_hat.permute(1, 0).reshape(1, -1, H, W).detach().cpu()
    if torch.max(feature_hat) > 128:
        max_val = 255
    else:
        max_val = 1
    if torch.min(feature_hat) < -0.5:
        min_val = -1
    else:
        min_val = 0
    L = max_val - min_val
    padd = 0
    _, channel, height, width = feature_hat.size()
    window = create_window(w_size, channel=channel).to(feature_hat.device)
    mu1 = F.conv2d(feature_hat, window, padding=padd, groups=channel)
    mu2 = F.conv2d(feature, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)                                                         # 均值mu1^2
    mu2_sq = mu2.pow(2)                                                         # 均值mu2^2
    mu1_mu2 = mu1 * mu2                                                         # mu1 * mu2
    sigma1_sq = F.conv2d(feature_hat * feature_hat, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(feature * feature, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(feature_hat * feature, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2                                                        # c1 = (0.01 * L)^2
    C2 = (0.03 * L) ** 2                                                        # c2 = (0.03 * L)^2
    v1 = 2.0 * sigma12 + C2                                                     # v1 = 2 * \sigma1 * \sigma2
    v2 = sigma1_sq + sigma2_sq + C2                                             # v2 = \sigma1^2 + \sigma2^2 + c2
    cs = torch.mean(v1 / v2)                        # contrast sensitivity      # cs = mean(v1 / v2)     对比度比较
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)        # 计算SSIM
    if size_average:                                                            # 如果计算均值
        ret = ssim_map.mean()                                                   # 计算SSIM均值
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret

def create_window(w_size, channel=1):
    _1D_window = gaussian(w_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
    return window

def gaussian(w_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
    return gauss/gauss.sum()      # $gauss = e^{-(x-5)^2/2\sigma^2}$

def cal_lpips(feature, feature_hat, H, W):
    feature = feature.permute(1, 0).reshape(1, -1, H, W).detach().cpu()
    feature_hat = feature_hat.permute(1, 0).reshape(1, -1, H, W).detach().cpu()
    lpips_fn = lpips.LPIPS(net='alex')
    lpipss = lpips_fn(feature, feature_hat)
    return lpipss
    
if __name__=="__main__":
    None