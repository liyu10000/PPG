import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bin_mask(mask):
    """ Check which side of mask contains bigger segments, combine the less to more.
    """
    if mask.shape[0] == 1 or mask.shape[2] == 1:
        return mask
    if mask.shape[2] == 6:
        mask = mask.transpose((2, 0, 1))
    
    stbd = np.sum(mask[:3, :, :])
    ps   = np.sum(mask[3:, :, :])
    if stbd > ps:
        mask[:3, :, :] = np.where(mask[3:, :, :] > 0, mask[3:, :, :], mask[:3, :, :])
        mask[3:, :, :].fill(0)
    else:
        mask[3:, :, :] = np.where(mask[:3, :, :] > 0, mask[:3, :, :], mask[3:, :, :])
        mask[:3, :, :].fill(0)
    return mask


def plt_mask(mask, save_name=None):
    """ plot mask with 1 or 3 or 6 channels
    """
    if mask.shape[2] == 6 or mask.shape[2] == 3 or mask.shape[2] == 1:
        mask = mask.transpose((2, 0, 1))

    if mask.shape[0] == 6:
        f = plt.figure(1, figsize=(18, 22))
        classes = ['STBD TS', 'STBD BT', 'STBD VS', 'PS TS', 'PS BT', 'PS VS']
        for i in range(3):
            plt.subplot(3, 2, 2*i+1)
            plt.imshow(mask[i, :, :], "gray")
            plt.xticks([])
            plt.yticks([])
            plt.title(classes[i], fontsize=40)
            plt.subplot(3, 2, 2*i+2)
            plt.imshow(mask[i+3, :, :], "gray")
            plt.xticks([])
            plt.yticks([])
            plt.title(classes[i+3], fontsize=40)
    elif mask.shape[0] == 3:
        f = plt.figure(1, figsize=(9, 22))
        classes = ['TS', 'BT', 'VS']
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.imshow(mask[i, :, :], "gray")
            plt.xticks([])
            plt.yticks([])
            plt.title(classes[i], fontsize=40)
    else:
        f = plt.figure(1, figsize=(9, 7))
        classes = ['SHIP']
        i = 0
        plt.imshow(mask[i, :, :], "gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(classes[i], fontsize=40)
    if save_name is not None:
        plt.savefig(save_name)
    f.clear()
    plt.close(f)
    # plt.show()


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
    train_log = './train.txt'
    epochs, train_losses, val_losses, train_ious, val_ious = parse_train_log(train_log)
    print(epochs)
    print(train_losses)
    print(val_ious)