import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def bin_mask(mask):
    """ Check which side of mask contains bigger segments, combine the less to more.
    """
    if mask.shape[0] == 1 or mask.shape[2] == 1:
        return mask
    if mask.shape[0] == 3 or mask.shape[2] == 3:
        return mask
    if mask.shape[2] == 6:  # convert from H,W,C to C,H,W
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


def plot_mask(mask, save_name=None):
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


def img_open(img):
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=3)
    return img

def img_close(img):
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def img_close2(img):
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def combine_contours(contours):
    if len(contours) <= 1:
        return contours
    c1, c2 = contours[0], contours[1]
    if cv2.contourArea(c2) / cv2.contourArea(c1) < 0.5:
        return [c1]
    c1y, c2y = c1[:, :, 1], c2[:, :, 1]
    c1_ymin = np.min(c1y, axis=0)
    c1_ymax = np.max(c1y, axis=0)
    c2_ymin = np.min(c2y, axis=0)
    c2_ymax = np.max(c2y, axis=0)
    c1_yctr = (c1_ymax + c1_ymin) / 2
    c2_yctr = (c2_ymax + c2_ymin) / 2
    c1_ydim = c1_ymax - c1_ymin
    c2_ydim = c2_ymax - c2_ymin
    if abs(c1_yctr - c2_yctr) < (c1_ydim + c2_ydim) / 2:
        contours = [c1, c2]
    else:
        contours = [c1]
    return contours

def contour(img):
    img = (-img.astype(np.uint8)) * 255
    # ret, img2 = cv2.threshold(img2, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = combine_contours(contours)
    img = np.zeros_like(img)
    cv2.drawContours(img, contours, -1, (255, 0, 0), -1)  # fill the biggest contour
    return img

def contour_convex(img):
    img = (-img.astype(np.uint8)) * 255
    # ret, img2 = cv2.threshold(img2, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    hull = cv2.convexHull(contours[0], False)
    img = np.zeros_like(img)
    cv2.drawContours(img, [hull], -1, (255, 0, 0), -1)  # fill convex hull
    return img

def get_3channelmask(mask):
    if mask.shape[0] == 3 or mask.shape[0] == 6:
        mask = mask.transpose((1, 2, 0))
    if mask.shape[2] == 3:
        return mask
    stbd = np.sum(mask[:, :, :3])
    if stbd == 0:
        return mask[:, :, 3:]
    else:
        return mask[:, :, :3]

def process_mask(mask):
    mask2 = np.zeros_like(mask)
    mask = mask.astype(np.float32)
    for i in range(3):
        img = mask[:, :, i]
        if np.sum(img) > 0:
            img = img_close2(img)
            img = contour(img)
            # img = contour_convex(img)
        mask2[:, :, i] = img
    return mask2

def plot_color(img):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_gray(img):
    plt.imshow(img, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_mask_on_img(img, mask, save_name=None):
    if np.max(mask) == 1:
        mask *= 255
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    img = cv2.addWeighted(img, 1.0, mask, 0.5, 0)
    if save_name is None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plot_color(img)
    else:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_name, img)