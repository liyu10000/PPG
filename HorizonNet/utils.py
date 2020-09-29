import torch
import torch.nn as nn
from collections import OrderedDict


def group_weight(module):
    # Group module parameters into two group
    # One need weight_decay and the other doesn't
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]


def adjust_learning_rate(optimizer, args):
    if args.cur_iter < args.warmup_iters:
        frac = args.cur_iter / args.warmup_iters
        step = args.lr - args.warmup_lr
        args.running_lr = args.warmup_lr + step * frac
    else:
        frac = (float(args.cur_iter) - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        scale_running_lr = max((1. - frac), 0.) ** args.lr_pow
        args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr


def save_model(net, path, ith_epoch, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.backbone,
            'use_rnn': net.use_rnn,
        },
        'ith_epoch': ith_epoch,
        'state_dict': net.state_dict(),
    })
    torch.save(state_dict, path)


def load_trained_model(Net, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(**state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return state_dict['ith_epoch'], net


def read_txt(seg_txt):
    bon = []
    with open(seg_txt, 'r') as f:
        for i, line in enumerate(f.readlines()):
            tokens = line.strip().split()
            if i == 0:
                H, W = int(tokens[1]), int(tokens[3])
                h, w = int(tokens[5]), int(tokens[7])
            elif i == 1:
                h1, w1 = int(tokens[1]), int(tokens[3])
                h2, w2 = int(tokens[5]), int(tokens[7])
            elif i == 2:
                w1_begin, w1_end = int(tokens[1]), int(tokens[3])
            elif i == 3:
                w2_begin, w2_end = int(tokens[1]), int(tokens[3])
            else:
                bon.append([float(tokens[0]), float(tokens[1])])
    txt_content = {'H':H, 'W':W, 'h':h, 'w':w,
                   'h1':h1, 'w1':w1, 'h2':h2, 'w2':w2, 
                   'w1_begin':w1_begin, 'w1_end':w1_end, 
                   'w2_begin':w2_begin, 'w2_end':w2_end, 
                   'bon':bon}
    return txt_content

def write_txt(seg_txt, txt_content):
    H, W = txt_content['H'], txt_content['W']
    h, w = txt_content['h'], txt_content['w']
    h1, w1 = txt_content['h1'], txt_content['w1']
    h2, w2 = txt_content['h2'], txt_content['w2']
    w1_begin, w1_end = txt_content['w1_begin'], txt_content['w1_end']
    w2_begin, w2_end = txt_content['w2_begin'], txt_content['w2_end']
    bon = txt_content['bon']
    with open(seg_txt, 'w') as f:
        # write H, W, h, w
        f.write('H: {} W: {} h: {} w: {}\n'.format(H, W, h, w))
        # write h1, w1, h2, w2 (cropping positions)
        f.write('h1: {} w1: {} h2: {} w2: {}\n'.format(h1, w1, h2, w2))
        # write start&end points of the two boundaries
        f.write('w1_begin: {} w1_end: {}\n'.format(w1_begin, w1_end))
        f.write('w2_begin: {} w2_end: {}\n'.format(w2_begin, w2_end))
        # write seg lines positions
        for h1, h2 in bon:
            f.write('{:.1f} {:.1f}\n'.format(h1, h2))

def write_pred_to_txt(pred_txt, bon):
    txt_content = {'H':0, 'W':0, 'h':0, 'w':0,
                   'h1':0, 'w1':0, 'h2':0, 'w2':0, 
                   'w1_begin':0, 'w1_end':0, 
                   'w2_begin':0, 'w2_end':0, 
                   'bon':bon}
    write_txt(pred_txt, txt_content)
