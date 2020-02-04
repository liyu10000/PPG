import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plt_6image(img):
    """ plot img with 6 channels
    """
    if img.shape[2] == 6:
        img = img.transpose((2, 0, 1))

    plt.figure(1, figsize=(18, 22))

    classes = ['STBD TS', 'STBD BT', 'STBD VS', 'PS TS', 'PS BT', 'PS VS']
    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.imshow(img[i, :, :], "gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(classes[i], fontsize=40)

        plt.subplot(3, 2, 2*i+2)
        plt.imshow(img[i+3, :, :], "gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(classes[i+3], fontsize=40)

    plt.show()


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