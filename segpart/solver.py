import os
import cv2
import time
import argparse
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .model import HorizonNet
from .data import generator
from .utils import adjust_learning_rate, save_model, load_trained_model, write_pred_to_txt



def masked_l1_loss(input, target, mask):
    loss = F.l1_loss(input, target, reduction='none') # l1 loss
    # loss = (input - target) ** 2 # l2 loss
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def feed_forward(net, x, y_bon, y_bon_mask, device):
    x = x.to(device)
    y_bon = y_bon.to(device)
    y_bon_mask = y_bon_mask.to(device)
    y_bon_pred = net(x)
    loss = F.l1_loss(y_bon_pred, y_bon)
    # loss = masked_l1_loss(y_bon_pred, y_bon, y_bon_mask)
    losses = {'total':loss}
    return losses

def segpart_train(cfg):
    device = torch.device('cuda')
    os.makedirs(cfg.ckpt, exist_ok=True)

    # Create dataloader
    loader_train = generator(cfg.img_dirs, 
                             cfg.txt_dirs,
                             phase='train',
                             train_val_split=cfg.train_val_split,
                             return_bon=True,
                             return_path=False,
                             batch_size=cfg.train_batch_size,
                             num_workers=cfg.num_workers)
    loader_valid = generator(cfg.img_dirs, 
                             cfg.txt_dirs,
                             phase='valid',
                             train_val_split=cfg.train_val_split,
                             return_bon=True,
                             return_path=False,
                             batch_size=cfg.val_batch_size,
                             num_workers=cfg.num_workers)

    # Create model
    if cfg.pth is not None:
        print('Finetune model is given:', cfg.pth)
        print('Ignore --backbone and --no_rnn')
        start_epoch, net = load_trained_model(HorizonNet, cfg.pth)
        net = net.to(device)
    else:
        net = HorizonNet(cfg.backbone, not cfg.no_rnn).to(device)
        start_epoch = 0

    # Create optimizer
    if cfg.optim == 'SGD':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    elif cfg.optim == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    else:
        raise NotImplementedError()

    # Init variable
    cfg.warmup_iters = cfg.warmup_epochs * len(loader_train)
    cfg.max_iters = cfg.epochs * len(loader_train)
    cfg.running_lr = cfg.warmup_lr if cfg.warmup_epochs > 0 else cfg.lr
    cfg.cur_iter = 0
    cfg.best_valid_score = float('inf')

    # Start training
    for ith_epoch in range(start_epoch + 1, cfg.epochs + start_epoch + 1):

        # Train phase
        net.train()
        train_loss = {}
        for x, y_bon, y_bon_mask in loader_train:
            # Set learning rate
            adjust_learning_rate(optimizer, cfg)

            cfg.cur_iter += 1

            losses = feed_forward(net, x, y_bon, y_bon_mask, device)
            loss = losses['total']
            for k, v in losses.items():
                train_loss[k] = train_loss.get(k, 0) + v.item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 3.0, norm_type='inf')
            optimizer.step()

        # Valid phase
        net.eval()
        valid_loss = {}
        for x, y_bon, y_bon_mask in loader_valid:
            with torch.no_grad():
                losses = feed_forward(net, x, y_bon, y_bon_mask, device)
            for k, v in losses.items():
                valid_loss[k] = valid_loss.get(k, 0) + v.item()

        # Save best validation loss model
        now_train_score = train_loss['total'] / len(loader_train)
        now_valid_score = valid_loss['total'] / len(loader_valid)
        cur_time = time.strftime("%H:%M:%S")
        print('Ep%3d | %s | Train %.4f | Valid %.4f vs. Best %.4f' % (ith_epoch, cur_time, now_train_score, now_valid_score, cfg.best_valid_score))
        if now_valid_score < cfg.best_valid_score:
            cfg.best_valid_score = now_valid_score

        # Periodically save model
        if ith_epoch % cfg.save_every == 0 or ith_epoch == cfg.epochs + start_epoch:
            save_model(net,
                       os.path.join(cfg.ckpt, '{}_{}.pth'.format(cfg.keyword, ith_epoch)),
                       ith_epoch)

def segpart_test(cfg):
    device = torch.device('cuda')
    os.makedirs(cfg.pred_dir, exist_ok=True)

    # Create dataloader
    loader_test = generator([cfg.test_img_dir], 
                            [cfg.test_txt_dir],
                            phase='test',
                            train_val_split='',
                            return_bon=cfg.calc_loss,
                            return_path=True,
                            batch_size=cfg.test_batch_size,
                            num_workers=cfg.test_num_workers)
    # Load model
    _, net = load_trained_model(HorizonNet, cfg.test_pth)
    net = net.to(device)
    print('Load model from ', os.path.basename(cfg.test_pth))

    # Start testing
    net.eval()
    colors = [(0,255,0), (0,0,255)]
    if cfg.calc_loss:
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
                    pred_mask = os.path.join(cfg.pred_dir, os.path.basename(path))
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
                    pred_mask = os.path.join(cfg.pred_dir, os.path.basename(path))
                    cv2.imwrite(pred_mask, img)
                    pred_txt = os.path.join(cfg.pred_dir, os.path.basename(path)[:-4]+'.txt')
                    write_pred_to_txt(pred_txt, bon.transpose((1, 0)))
                    print('saved ', pred_mask)