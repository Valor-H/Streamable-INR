import torch
import argparse
from train import ImageTrainer

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Training hyperparameter configuration")
    parser.add_argument("--mode", type=str, default="coin", choices=["coin, sw, sd, swd"], help="Training model selection")
    parser.add_argument("--state", type=str, default="train", choices=["train, meta, mtrain, quant_entropy"], help="Training state selection")
    parser.add_argument("--widths", type=int, default=[30, 60, 90] , help="Model width configuration")
    parser.add_argument("--depths", type=int, default=[3, 4, 5], help="Model depth configuration")
    parser.add_argument("--lr", type=float, default=2e-4, help="Training learning rate")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--data_path", type=str, default="./data/Kodak", help="Training data path")
    ## LOGS
    parser.add_argument("--logs_path", type=str, default="./logs", help="Log saving path")
    parser.add_argument("--logs_inter", type=int, default=100, help="Log saving interval")
    parser.add_argument("--save_inter", type=int, default=100, help="Image and model save interval")
    ## MAML
    parser.add_argument("--meta_path", type=str, default="./data/Kodak", help="Meta-learning training data path")
    parser.add_argument("--out_epochs", type=int, default=4, help="The number of repetitions in the dataset")
    parser.add_argument("--in_epochs", type=int, default=3, help="The number of iterations inner loop training")
    parser.add_argument("--out_lr", type=float, default=5e-5, help="Outer loop learning rate")
    parser.add_argument("--in_lr", type=float, default=1e-2, help="Inner loop learning rate")
    parser.add_argument("--lr_type", type=str, default="step_param", choices=["static, param, step_param"], help="Learning rate type")
    ## QUANT_ENTROPY
    parser.add_argument("--num_bits", type=int, default=[14], help="Quantization bit-width")
    parser.add_argument("--std_range", type=int, default=14, help="Quantization scaling standard deviation range")
    parser.add_argument("--is_entropy", type=bool, default=True, help="Entropy coding")
    parser.add_argument("--plot_distribution", type=bool, default=False, help="Plot the parameter distribution")
    ## OTHER
    parser.add_argument("--eval_all", type=bool, default=True, help="Evaluate only one metric: PSNR [LPIPS MS-SSIM]")
    parser.add_argument("--img_name", type=str, default=["kodim15", "kodim20"], help="Training image list")
    parser.add_argument("--reshape", type=int, default=[512, 768], help="Reshape images facilitate meta-learning")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--cuda", type=str, default="1", help="cuda")

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