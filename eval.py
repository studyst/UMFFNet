import os
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
# from skimage.measure import compare_ssim
import time
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def transformer(image, label):
    # print(image.size(2))
    H = image.size(2)
    W = image.size(3)
    if H % 2 == 1:
        H = H - 1
    if W % 2 == 1:
        W = W - 1

    cut = torchvision.transforms.CenterCrop([W, H])
    to_tensor=torchvision.transforms.ToTensor()
    image = to_tensor(cut(image))
    label = to_tensor(cut(label))

    return image, label

def _eval(generator, args):
    print(args.test_model)
    state_dict = torch.load(args.test_model) # cpu
    generator.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    generator.eval()
    # bayes_cap.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()


        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            # input_img, label_img = transformer(input_img, label_img)

            input_img = input_img.to(device)

            tm = time.time()

            # pred = model(input_img)[2]
            # print(name)
            pred,alpha,beta = generator(input_img) # ,alpha,beta
            pred = pred[2]
            # pred,_,_ = bayes_cap(pred)
            # pred = model(input_img, input_img, training=False)[2] #un

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            # a_map = (1 / (alpha[0] + 1e-5)).to('cpu').data
            # # print(pred_alpha)
            # plt.imshow(a_map.squeeze(), cmap='Blues') # Blues
            # # plt.imshow(a_map.transpose(0, 2).transpose(0, 1), cmap='inferno')
            # plt.clim(0, 0.05)
            # plt.axis('off')
            # plt.savefig('./results/UMFFNetPlus/alpha/' + name[0], bbox_inches='tight', pad_inches=0)
            #
            # beta = beta.to('cpu').data
            # # print(pred_alpha)
            # plt.imshow(beta.squeeze(), cmap='gray')  # Blues
            # plt.clim(0.45, 0.75)
            # plt.axis('off')
            # plt.savefig('./results/UMFFNetPlus/beta/' + name[0], bbox_inches='tight', pad_inches=0)
            #
            #
            # u_map = (a_map ** 2) * (
            #         torch.exp(torch.lgamma(3 / (beta + 1e-2))) / torch.exp(torch.lgamma(1 / (beta + 1e-2))))
            # # print(u_map)
            # plt.imshow((u_map).squeeze(), cmap='gist_heat')
            # plt.clim(0, 0.10)
            # plt.axis('off')
            # # plt.colorbar()
            # plt.savefig('./results/UMFFNetPlus/un/' + name[0], bbox_inches='tight', pad_inches=0)

            # pred = pred.cpu()
            # label_img = label_img.cpu()
            # print(pred.shape,label_img.shape)
            # error_map = torch.mean(torch.pow(torch.abs(pred - label_img), 2), dim=0).to('cpu').data
            # # plt.imshow((error_map).squeeze(), cmap='jet')
            # plt.imshow(error_map, cmap='jet')
            # plt.clim(0, 1)
            # plt.axis('off')
            # plt.savefig('./results/UMFFNetPlus/error/' + name[0], bbox_inches='tight', pad_inches=0)

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

                # save_name2 = os.path.join('./results/UMFFNetPlus/beta1', name[0])
                # beta += 0.5 / 255
                # beta = F.to_pil_image(beta.squeeze(0).cpu())
                # beta.save(save_name2)

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)

            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print("Average time: %f" % adder.average())