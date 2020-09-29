import os
import cv2
import argparse
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from model import HorizonNet, ENCODER_RESNET, ENCODER_DENSENET
from dataset import PPGPartDataset, generator
from utils import load_trained_model, write_pred_to_txt


def masked_l1_loss(input, target, mask):
    loss = F.l1_loss(input, target, reduction='none') # l1 loss
    # loss = (input - target) ** 2 # l2 loss
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', default=None,
                        help='path to load saved checkpoint. (testing)')
    parser.add_argument('--calc_loss', action='store_true', help='calculate test loss')
    # Dataset related arguments
    parser.add_argument('--img_dirs', action='append', help='path/to/images')
    parser.add_argument('--txt_dirs', action='append', help='path/to/txts')
    parser.add_argument('--pred_dir', default='path/to/save/predictions')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='numbers of workers for dataloaders')
    # optimization related arguments
    parser.add_argument('--batch_size_test', default=4, type=int,
                        help='training mini-batch size')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    parser.add_argument('--seed', default=42, type=int,
                        help='manual seed')

    args = parser.parse_args()
    device = torch.device('cpu' if args.no_cuda else 'cuda')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.pred_dir, exist_ok=True)

    # Create dataloader
    loader_test = generator(args.img_dirs, 
                            args.txt_dirs,
                            phase='test',
                            seed=args.seed,
                            train_val_split='',
                            return_bon=args.calc_loss,
                            return_path=True,
                            batch_size=args.batch_size_test,
                            num_workers=args.num_workers)
    # Load model
    _, net = load_trained_model(HorizonNet, args.pth)
    net = net.to(device)
    print('Load model from ', os.path.basename(args.pth))

    # Start testing
    net.eval()
    colors = [(0,255,0), (0,0,255)]
    if args.calc_loss:
        test_loss = 0
        for imgs, y_bons, y_bon_masks, paths in loader_test:
            imgs = imgs.to(device)
            y_bons = y_bons.to(device)
            y_bon_masks = y_bon_masks.to(device)
            with torch.no_grad():
                bons = net(imgs)
                loss = masked_l1_loss(bons, y_bons, y_bon_masks)
                test_loss += loss.item()
                imgs = imgs.detach().cpu().numpy().transpose((0, 2, 3, 1)).copy()
                imgs = (imgs * 255).astype(np.uint8)
                B, H, W, C = imgs.shape
                bons = bons.detach().cpu().numpy()
                bons = (bons * H / 4).astype(np.int32)
                for img, bon, path in zip(imgs, bons, paths):
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    for i in range(2):
                        for x,y in enumerate(bon[i]):
                            img = cv2.circle(img, (x, y), radius=3, color=colors[i], thickness=-1)
                    pred_mask = os.path.join(args.pred_dir, os.path.basename(path))
                    cv2.imwrite(pred_mask, img)
                    print('saved ', pred_mask)
        test_loss /= len(loader_test)
        print('Test loss: {:.4f}'.format(test_loss))
    else:
        for imgs, paths in loader_test:
            imgs = imgs.to(device)
            with torch.no_grad():
                bons = net(imgs)
                imgs = imgs.detach().cpu().numpy().transpose((0, 2, 3, 1)).copy()
                imgs = (imgs * 255).astype(np.uint8)
                B, H, W, C = imgs.shape
                imgs = np.zeros((B, H, W, C), dtype=np.uint8)
                bons = bons.detach().cpu().numpy()
                bons = (bons * H / 4).astype(np.int32)
                for img, bon, path in zip(imgs, bons, paths):
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    for i in range(2):
                        for x,y in enumerate(bon[i]):
                            img = cv2.circle(img, (x, y), radius=3, color=colors[i], thickness=-1)
                    pred_mask = os.path.join(args.pred_dir, os.path.basename(path))
                    cv2.imwrite(pred_mask, img)
                    pred_txt = os.path.join(args.pred_dir, os.path.basename(path)[:-4]+'.txt')
                    write_pred_to_txt(pred_txt, bon.transpose((1, 0)))
                    print('saved ', pred_mask)
