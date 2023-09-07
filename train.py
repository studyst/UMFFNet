import os
import torch

from data import train_dataloader
from utils import Adder, Timer, check_lr
#from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import math
import torch.optim as optim
import torch.nn.functional as F
from improve_utils.loss import *


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#warm_up
class WarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, gamma, last_epoch=-1):
        """
        optimizer: 优化器对象
        warmup_steps: 学习率线性增加的步数
        gamma: 学习率下降系数
        last_epoch: 当前训练轮数
        """
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 学习率线性增加
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # 学习率按指数衰减
            return [base_lr * math.exp(-(self.last_epoch - self.warmup_steps + 1) * self.gamma) for base_lr in
                    self.base_lrs]


def _train(model, args):
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Cri = UncertaintyLoss(alpha_eps=1e-4, beta_eps=1e-4, resi_min=1e-4, resi_max=1e3)
    criterion = torch.nn.L1Loss()
    #warm_up
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = WarmupLR(optimizer, warmup_steps=50, gamma=args.gamma)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    # epoch = args.start_iter
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    # writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    iter_un_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img, alpha, beta= model(input_img) # , alpha, beta
            # pred_img = model(input_img, label_img, training=True) # un
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1+l2+l3



            label_fft1 = torch.fft.fft2(label_img4, dim=(-2, -1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)
            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2, -1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)
            label_fft2 = torch.fft.fft2(label_img2, dim=(-2, -1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)
            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2, -1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)
            label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)
            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2, -1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            loss_un = Cri(pred_img[2], alpha, beta, pred_img[2], label_img, T1=1, T2=5e-2)

            loss = loss_content + 0.1 * loss_fft + loss_un
            loss.backward()
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())


            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val_gopro = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average GOPRO PSNR %.2f dB' % (epoch_idx, val_gopro))
            if val_gopro >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)