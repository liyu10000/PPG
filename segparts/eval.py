import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import scan_files, get_label_dict, resize_with_pad, make_mask
from post_process import bin_mask, process_mask


def get_sidemask(mask):
    if mask.shape[0] == 3 or mask.shape[0] == 6:
        mask = mask.transpose((1, 2, 0))
    if mask.shape[2] == 3:
        return 'both', mask
    stbd = np.sum(mask[:, :, :3])
    if stbd == 0:
        return 'ps', mask[:, :, 3:]
    else:
        return 'stbd', mask[:, :, :3]

# PyTroch version
# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    # expect outputs to be H x W
    intersection = (outputs & labels).float().sum()
    union = (outputs | labels).float().sum()  
    iou = (intersection + SMOOTH) / (union + SMOOTH) 
    return iou

def calc_loss(image_dir, label_dir, pred_mask_dir, process_fn, dtype, loss_fn):
    label_dict = get_label_dict(image_dir, label_dir)
    class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 
                   'PS TS':3, 'PS BT':4, 'PS VS':5}
    names = [os.path.splitext(f)[0] for f in os.listdir(pred_mask_dir) 
                                    if f.endswith('.npy')]
    losses = [0.0] * 4 # TS, BT, VS, and all
    for name in names:
        print('processing', name)
        label_info = label_dict[name]
        img = cv2.imread(label_info["path"])
        img, factor, direction, pad = resize_with_pad(img)

        mask = make_mask(label_info, class_index, factor, direction, pad)
        mask = mask.astype(dtype) # C, H, W
        side, mask = get_sidemask(mask) # H, W, 3
        # print(img.shape, mask.shape)

        pred_mask = np.load(os.path.join(pred_mask_dir, name+'.npy'))
        pred_mask = pred_mask > 0.5  # convert to binary mask
        pred_mask = pred_mask.astype(np.uint8)
        if len(set(class_index.values())) == 6:
            pred_mask = bin_mask(pred_mask)
        pred_side, pred_mask = get_sidemask(pred_mask)
        pred_mask = process_fn(pred_mask)
        pred_mask = pred_mask / 255.
        pred_mask = pred_mask.astype(dtype)
        # print(pred_mask.shape)

        # print(name, side, pred_side)

        mask = torch.from_numpy(mask)
        pred_mask = torch.from_numpy(pred_mask)
        for i in range(3):
            loss = loss_fn(pred_mask[:, :, i], mask[:, :, i])
            losses[i] += loss
        loss = loss_fn(pred_mask, mask)
        losses[3] += loss

    n = len(names)
    losses = [loss / n for loss in losses]
    print(losses)


def parse_train_log(train_log):
    epochs = []
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    with open(train_log, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('Starting') and 'train' in line:  # train starts
                tokens = line.strip().split()
                epoch = int(tokens[2])
                epochs.append(epoch)  # number of epochs

                i += 1
                line = lines[i]
                tokens = line.strip().split()
                loss = float(tokens[1])
                iou = float(tokens[4])
                train_losses.append(loss)  # loss
                train_ious.append(iou)  # iou
            if line.startswith('Starting') and 'val' in line:  # val starts
                i += 1
                line = lines[i]
                tokens = line.strip().split()
                loss = float(tokens[1])
                iou = float(tokens[4])
                val_losses.append(loss)  # loss
                val_ious.append(iou)  # iou
            i += 1
    return epochs, train_losses, val_losses, train_ious, val_ious


def plot_train_log(epochs, train_metric, val_metric):
    # plot trend
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    ax.grid()
    ax.scatter(epochs, train_metric, marker='.', color='black')
    ax.scatter(epochs, val_metric, marker='s', color='red')

    plt.show()



if __name__ == '__main__':
    # # test train log plot
    # train_log = './train.txt'
    # epochs, train_losses, val_losses, train_ious, val_ious = parse_train_log(train_log)
    # print(epochs)
    # print(train_losses)
    # print(val_ious)

    # calculate bce loss for each side, each part
    # image_dir = "../data/labeled/images"
    # label_dir = "../data/labeled/labels"
    # pred_mask_dir = "../data/labeled/pred_masks/exp_0304/6classes/xcep_tv_90th"

    image_dir = "../data/Segmentation_Test_Set/images"
    label_dir = "../data/Segmentation_Test_Set/labels"
    pred_mask_dir = "../data/Segmentation_Test_Set/pred_masks"

    calc_loss(image_dir, label_dir, pred_mask_dir, process_mask, np.float32, nn.BCEWithLogitsLoss())
    calc_loss(image_dir, label_dir, pred_mask_dir, process_mask, np.uint8, iou_pytorch)