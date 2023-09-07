import os
import torch
import argparse
from torch.backends import cudnn
from models.UMFFNet import build_net
from train import _train
from eval import _eval
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)
    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='UMFFNet', choices=['UMFFNet', 'UMFFNetPlus'], type=str)
    parser.add_argument('--data_dir', type=str, default='datasets/train/MixTrain')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--start_iter', type=int, default=20, help='Starting Epoch')
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10000)
    parser.add_argument('--resume', type=str, default='results/UMFFNet/weights/model.pkl')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 50 for x in range(300//50)])
    parser.add_argument('--seed', type=int, default=3407, help='random seed to use. Default=123')

    # Test
    parser.add_argument('--test_model', type=str, default='weights/UMFFNet.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')
    print(args)
    main(args)
