import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model import HorizonNet, ENCODER_RESNET, ENCODER_DENSENET
from dataset import PPGPartDataset, generator
from utils import group_weight, adjust_learning_rate, save_model, load_trained_model



def masked_l1_loss(input, target, mask):
    loss = F.l1_loss(input, target, reduction='none') # l1 loss
    # loss = (input - target) ** 2 # l2 loss
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def feed_forward(net, x, y_bon, y_bon_mask):
    x = x.to(device)
    y_bon = y_bon.to(device)
    y_bon_mask = y_bon_mask.to(device)
    y_bon_pred = net(x)
    # loss = F.l1_loss(y_bon_pred, y_bon)
    loss = masked_l1_loss(y_bon_pred, y_bon, y_bon_mask)
    losses = {'total':loss}
    return losses


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id', required=True,
                        help='experiment id to name checkpoints and logs')
    parser.add_argument('--keyword', required=True, help='basename for checkpoints')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--pth', default=None,
                        help='path to load saved checkpoint. (finetuning)')
    # Model related
    parser.add_argument('--backbone', default='resnet50',
                        choices=ENCODER_RESNET + ENCODER_DENSENET,
                        help='backbone of the network')
    parser.add_argument('--no_rnn', action='store_true',
                        help='whether to remove rnn or not')
    # Dataset related arguments
    parser.add_argument('--names_file', type=str, default='', help='csv file containing names list')
    parser.add_argument('--img_dirs', action='append', help='path/to/images')
    parser.add_argument('--txt_dirs', action='append', help='path/to/txts')
    parser.add_argument('--train_val_split', type=str, help='split/in/n,i')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='numbers of workers for dataloaders')
    # optimization related arguments
    parser.add_argument('--freeze_earlier_blocks', default=-1, type=int)
    parser.add_argument('--batch_size_train', default=4, type=int,
                        help='training mini-batch size')
    parser.add_argument('--batch_size_valid', default=2, type=int,
                        help='validation mini-batch size')
    parser.add_argument('--epochs', default=300, type=int,
                        help='epochs to train')
    parser.add_argument('--optim', default='Adam',
                        help='optimizer to use. only support SGD and Adam')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--warmup_lr', default=1e-6, type=float,
                        help='starting learning rate for warm up')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='numbers of warmup epochs')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='factor for L2 regularization')
    parser.add_argument('--bn_momentum', type=float)
    # Misc arguments
    parser.add_argument('--gpu', type=int, default=0, 
                        help='choose id of gpu to use')
    parser.add_argument('--seed', default=42, type=int,
                        help='manual seed')
    parser.add_argument('--disp_iter', type=int, default=1,
                        help='iterations frequency to display')
    parser.add_argument('--save_every', type=int, default=30,
                        help='epochs frequency to save state_dict')
    args = parser.parse_args()
    print(args)
    if args.gpu > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.ckpt, args.id), exist_ok=True)

    # Create dataloader
    df = pd.read_csv(args.names_file)
    df = df[df.train == 1]
    names = df.name.to_list()
    print(len(names), names)
    loader_train = generator(names,
                             args.img_dirs, 
                             args.txt_dirs,
                             phase='train',
                             seed=args.seed,
                             train_val_split=args.train_val_split,
                             return_bon=True,
                             return_path=False,
                             batch_size=args.batch_size_train,
                             num_workers=args.num_workers)
    loader_valid = generator(names,
                             args.img_dirs, 
                             args.txt_dirs,
                             phase='valid',
                             seed=args.seed,
                             train_val_split=args.train_val_split,
                             return_bon=True,
                             return_path=False,
                             batch_size=args.batch_size_valid,
                             num_workers=args.num_workers)

    # Create model
    if args.pth is not None:
        print('Finetune model is given:', args.pth)
        print('Ignore --backbone and --no_rnn')
        start_epoch, net = load_trained_model(HorizonNet, args.pth)
        net = net.to(device)
    else:
        net = HorizonNet(args.backbone, not args.no_rnn).to(device)
        start_epoch = 0

    assert -1 <= args.freeze_earlier_blocks and args.freeze_earlier_blocks <= 4
    if args.freeze_earlier_blocks != -1:
        b0, b1, b2, b3, b4 = net.feature_extractor.list_blocks()
        blocks = [b0, b1, b2, b3, b4]
        for i in range(args.freeze_earlier_blocks + 1):
            print('Freeze block%d' % i)
            for m in blocks[i]:
                for param in m.parameters():
                    param.requires_grad = False

    if args.bn_momentum:
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = args.bn_momentum

    # Create optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    # Init variable
    args.warmup_iters = args.warmup_epochs * len(loader_train)
    args.max_iters = args.epochs * len(loader_train)
    args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
    args.cur_iter = 0
    args.best_valid_score = float('inf')

    # Start training
    for ith_epoch in range(start_epoch + 1, args.epochs + start_epoch + 1):

        # Train phase
        net.train()
        if args.freeze_earlier_blocks != -1:
            blocks = net.feature_extractor.list_blocks() # b0, b1, b2, b3, b4
            for i in range(args.freeze_earlier_blocks + 1):
                for m in blocks[i]:
                    m.eval()
        train_loss = {}
        for x, y_bon, y_bon_mask in loader_train:
            # Set learning rate
            adjust_learning_rate(optimizer, args)

            args.cur_iter += 1

            losses = feed_forward(net, x, y_bon, y_bon_mask)
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
                losses = feed_forward(net, x, y_bon, y_bon_mask)
            for k, v in losses.items():
                valid_loss[k] = valid_loss.get(k, 0) + v.item()

        # Save best validation loss model
        now_train_score = train_loss['total'] / len(loader_train)
        now_valid_score = valid_loss['total'] / len(loader_valid)
        cur_time = time.strftime("%H:%M:%S")
        print('Ep%3d | %s | Train %.4f | Valid %.4f vs. Best %.4f' % (ith_epoch, cur_time, now_train_score, now_valid_score, args.best_valid_score))
        if now_valid_score < args.best_valid_score:
            args.best_valid_score = now_valid_score
            # if ith_epoch > args.save_every: # skip the first few epochs
            #     save_model(net,
            #                os.path.join(args.ckpt, args.id, '{}_best.pth'.format(args.keyword)), 
            #                ith_epoch
            #                args)

        # Periodically save model
        if ith_epoch % args.save_every == 0 or ith_epoch == args.epochs + start_epoch:
            save_model(net,
                       os.path.join(args.ckpt, args.id, '{}_{}.pth'.format(args.keyword, ith_epoch)),
                       ith_epoch,
                       args)

