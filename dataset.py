import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms

""" 数据集 """
class Dataset():
    """ 初始化 """
    def __init__(self, data_path, reshape=None, device=None):
        self.data_path = data_path
        self.file_name = os.listdir(self.data_path)
        self.file_name.sort(key=lambda x:x[-10:-4])
        self.img_type = self.file_name[0][-4:]
        if reshape == None:
            self.transform = transforms.ToTensor()
        else:
            self.reshape = reshape
            self.transform = self.reshape_transfrom
        self.device = device

    """ 获取单张图片 """
    def __getitem__(self, index):
        img_name = self.file_name[index]
        img = Image.open(os.path.join(self.data_path, img_name))
        img = self.transform(img).to(self.device)
        C, H, W = img.shape[0], img.shape[1], img.shape[2]
        coord = self.make_grid(H, W).to(self.device)
        feature = img.reshape(C, -1).permute(1, 0)
        return {"coord": coord, "feature": feature, "name": img_name[:-4], "H": H, "W": W}

    """ 获取数据集长度 """
    def __len__(self):
        return len(self.file_name)

    """ 改变图片形状 """
    def reshape_transfrom(self, img):
        pad1, pad2 = 0, 0
        img_H, img_W = img.size[1], img.size[0]
        if img_H < img_W:
            crop_H, crop_W = self.reshape[0], self.reshape[1]
        else:
            crop_H, crop_W = self.reshape[1], self.reshape[0]
        if img_W < crop_W:
            pad1 = int((crop_W-img_W)/2)
        if img_H < crop_H:
            pad2 = int((crop_H-img_H)/2)
        crop_shape, padding = (crop_H, crop_W), (pad1, pad2)
        transform = transforms.Compose([transforms.ToTensor(), torchvision.transforms.RandomCrop(crop_shape, padding)])
        return transform(img)

    """ 通过路径获取单张图片 """
    def get_by_path(self, path, file_name):
        img = Image.open(os.path.join(path, file_name))
        img = self.transform(img).to(self.device)
        C, H, W = img.shape[0], img.shape[1], img.shape[2]
        coord = self.make_grid(H, W).to(self.device)
        feature = img.reshape(C, -1).permute(1, 0)
        return {"coord": coord, "feature": feature, "name": file_name[:-4], "H": H, "W": W}

    """ 获取图像坐标 """
    def make_grid(self, H, W):
        coords_x = torch.linspace(-1, 1, H)
        coords_y = torch.linspace(-1, 1, W)
        grid = torch.stack(torch.meshgrid(coords_x, coords_y), -1)
        return grid.reshape(-1, 2)

""" 测试 """
if __name__=="__main__":
    dataset = Dataset(data_path="../data/kodak")
    for i, img_data in enumerate(dataset):
        coord, feature, name, H, W = img_data["coord"], img_data["feature"], img_data["name"], img_data["H"], img_data["W"]
        print(i, coord.shape, feature.shape, name, H, W)
        feature = feature.permute(1, 0).reshape(-1, H, W)
        torchvision.utils.save_image(feature, os.path.join("./temp/", f'{name}.png'))