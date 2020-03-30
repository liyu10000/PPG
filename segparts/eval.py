import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import scan_files, get_label_dict, resize_with_pad, resize_without_pad, make_mask_with_pad, make_mask_without_pad
from post_process import bin_mask, process_mask
from config import Config


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
    """ Calculate IOU per channel
    :param outputs: predicted mask, H x W
    :param labels: ground truth mask, H x W
    """
    SMOOTH = 1e-6
    intersection = (outputs & labels).float().sum()
    union = (outputs | labels).float().sum()  
    iou = (intersection + SMOOTH) / (union + SMOOTH) 
    return iou

def calc_loss(cfg, process_fn, dtype, loss_fn):
    classes = cfg.classes
    image_dir = cfg.test_image_dir
    label_dir = cfg.test_label_dir
    pred_mask_dir = cfg.pred_mask_dir
    W = cfg.W
    H = cfg.H
    topad = True if cfg.resize_with_pad == 'True' else False

    label_dict = get_label_dict(image_dir, label_dir)
    if classes == 6:
        class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':3, 'PS BT':4, 'PS VS':5}
    elif classes == 3:
        class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':0, 'PS BT':1, 'PS VS':2}
    else:  # classes = 1
        class_index = {'STBD TS':0, 'STBD BT':0, 'STBD VS': 0, 'PS TS':0, 'PS BT':0, 'PS VS':0}
    names = [os.path.splitext(f)[0] for f in os.listdir(pred_mask_dir) 
                                    if f.endswith('.npy')]
    losses = [0.0] * 4 # TS, BT, VS, and all
    print('# files to evaluate', len(names))
    for name in names:
        # print('processing', name)
        label_info = label_dict[name]
        img = cv2.imread(label_info["path"])
        if topad:
            img, factor, direction, padding = resize_with_pad(img, W, H)
            mask = make_mask_with_pad(label_info, class_index, factor, direction, padding, W, H)
        else:
            img, w_factor, h_factor = resize_without_pad(img, W, H)
            mask = make_mask_without_pad(label_info, class_index, w_factor, h_factor, W, H)
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
        if np.max(pred_mask) == 255:
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


if __name__ == '__main__':
    # calculate bce loss for each side, each part
    cfg = Config().parse()

    calc_loss(cfg, process_mask, np.float32, nn.BCEWithLogitsLoss())
    calc_loss(cfg, process_mask, np.uint8, iou_pytorch)