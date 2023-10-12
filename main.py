import torch
import argparse
from train import ImageTrainer

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="训练超参数配置")
    parser.add_argument("--mode", type=str, default="swd", choices=["coin, sw, sd, swd"], help="training mode")                                  # V
    parser.add_argument("--state", type=str, default="demo", choices=["train, meta, mtrain, quant"], help="training state")
    parser.add_argument("--widths", type=int, default=[30, 60, 90] , help="model widths")               ##TODO                                            # V
    parser.add_argument("--depths", type=int, default=[2, 4, 6], help="model depths")                   ##TODO                                              # V
    parser.add_argument("--lr", type=float, default=2e-4, help="training learning rate")
    parser.add_argument("--epochs", type=int, default=10000, help="number of training cycles")
    parser.add_argument("--data_path", type=str, default="./data/kodak", help="training data path")
    parser.add_argument("--logs_path", type=str, default="./logs/Weight_Result_no", help="logs paths")  ##TODO                                               # V
    parser.add_argument("--logs_inter", type=int, default=100, help="logs interval")                                                            # V
    parser.add_argument("--save_inter", type=int, default=100, help="logs interval")                                                            # V
    ## MAML
    parser.add_argument("--meta_path", type=str, default="../data/CelebA", help="meta data path")                                               # V
    parser.add_argument("--out_epochs", type=int, default=4, help="out loop epochs")                                                            # V
    parser.add_argument("--in_epochs", type=int, default=3, help="in loop epochs")                                                              # V
    parser.add_argument("--out_lr", type=float, default=5e-5, help="out loop lr")                                                               # V
    parser.add_argument("--in_lr", type=float, default=1e-2, help="in loop lr")                                                                 # V
    parser.add_argument("--lr_type", type=str, default="step_param", choices=["static, param, step_param"], help="lr type")                     # V
    parser.add_argument("--save_idx", type=int, default=0, help="in loop lr")
    ## OTHER
    parser.add_argument("--eval_all", type=bool, default=True, help="PSNR [LPIPS MS-SSIM]")
    parser.add_argument("--img_name", type=str, default=["kodim15"], help="只训练列表里的图片")
    parser.add_argument("--reshape", type=int, default=None, help="尺度不变Reshape [512, 768], 不够则补全")                                                   # V
    parser.add_argument("--seed", type=int, default=100, help="random seeds")
    parser.add_argument("--cuda", type=str, default="6", help="cuda")
    parser.add_argument("--num_bits", type=int, default=[14], help="量化位数")
    parser.add_argument("--std_range", type=int, default=14, help="量化缩放标准差范围")
    parser.add_argument("--is_entropy", type=bool, default=True, help="是否熵编码")
    parser.add_argument("--plot_distribution", type=bool, default=False, help="是否绘制参数分布图")
    args = parser.parse_args()
    print("ARGS: ", args)

    ## CONFIG
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method("spawn")

    ## TRAINING
    print(f"--- Here is mode of {args.mode}  state of {args.state} ---")
    trainer = ImageTrainer(args)
    if args.mode in ["coin"]:
        for idx in range(0, len(args.widths)):
            trainer.coin_train(idx)
    elif args.mode in ["sw"]:
        for idx in range(0, len(args.widths)):
            trainer.sw_train(idx)
    elif args.mode in ["sd"]:
        for idx in range(0, len(args.widths)):
            trainer.sd_train(idx)
    elif args.mode in ["swd"]:
        for idx in range(0, len(args.widths)):
            trainer.swd_train(idx)