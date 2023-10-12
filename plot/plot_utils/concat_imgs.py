import os
from PIL import Image
def long_img(imgs_path):
    cur_shape = (6000, 1000)
    root = "/home/xch/codes/meta-swd/plot_imgs/weight_imgs/"
    imgs_name = ["no_init_2.jpg", "no_init_4.jpg", "no_init_6.jpg", "init_4.jpg", "init_6.jpg"]
    imgs, images = [], []
    # 获取当前文件夹中的所有jpg图像
    # imgs = [Image.open(fn) for fn in os.listdir(imgs_path) if fn.endswith('.jpg')]
    for img_name in imgs_name:
        imgs.append(os.path.join(root, img_name))
    print("======", imgs)

    for img in imgs:
        image = Image.open(img)
        images.append(image)
    print(images)

    # 单幅图像尺寸
    width, height = images[0].size
    width, height = images[0].resize(cur_shape).size

    # 创建空白长图
    result = Image.new(images[0].mode, (width, height*len(images)), color=0)    # 默认填充一张图片的像素值是黑色
    print(type(result), result.size)   # <class 'PIL.Image.Image'> (500, 3000)

    # 拼接
    for i, img in enumerate(images):
        img = img.resize(cur_shape)
        result.paste(img, box=(0, i*height))   # 把每一张img粘贴到空白的图中，注意，如果图片的宽度大于空白图的长度

    result.save("./plot_imgs/weight.jpg")

if __name__ == "__main__":
    long_img("plot/plot_imgs/")