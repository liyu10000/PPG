import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from data import scan_files, get_label_dict, resize_with_pad, resize_without_pad, make_mask_with_pad, make_mask_without_pad
from config import Config


def refine_mask(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

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

def img_close(img, i):
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    if i < 0: # standard close
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
    else: # close and open
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
    img = img / 255
    return img

def contour_convex(img):
    img = (-img.astype(np.uint8)) * 255
    # ret, img2 = cv2.threshold(img2, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    hull = cv2.convexHull(contours[0], False)
    img = np.zeros_like(img)
    cv2.drawContours(img, [hull], -1, (255, 0, 0), -1)  # fill convex hull
    img = img / 255
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

def get_lowerbound(mask):  # usually the second mask
    """ Get lower bound of mask, input should be 2d: H x W
    """
    H, W = mask.shape
    line_mask = np.sum(mask, axis=0) > 0
    bound = np.zeros((np.sum(line_mask), 2))
    b = 0
    for i in range(W):
        if line_mask[i]:
            j = H - 1 - np.argmax(mask[:, i][::-1])
            # if j < H - 3: # remove bound on image border
            bound[b][0] = i
            bound[b][1] = j
            b += 1
    return bound

# def finetune_lowerbound(bound, lowermask):
#     H, W = lowermask.shape
#     line_mask = np.sum(lowermask, axis=0) > 0
#     bound = [bound[i] for ]

def get_upperbound(mask):  # usually the second mask
    """ Get upper bound of mask, input should be 2d: H x W
    """
    H, W = mask.shape
    line_mask = np.sum(mask, axis=0) > 0
    bound = np.zeros((np.sum(line_mask), 2))
    b = 0
    for i in range(W):
        if line_mask[i]:
            j = np.argmax(mask[:, i])
            # if j > 2: # remove bound on image border
            bound[b][0] = i
            bound[b][1] = j
            b += 1
    return bound

def remove_outliers(data):
    y = data[:, 1]
    threshold = 3
    mean = np.mean(y)
    std = np.std(y)
    
    data_new = []
    for i,d in enumerate(y):
        z_score= (d - mean) / std 
        if abs(z_score) < threshold:
            data_new.append(data[i, :])
    return np.array(data_new)

def zero_pad(mask, factor, direction, pad):
    """ zero pad at predicted mask
    :param mask: H, W, C
    :param factor: scaling factor
    :param direction: pad direction, Height or Width
    :param pad: pad size
    """
    if pad > 0:
        if direction == "Height":
            mask[:pad, :, :] = 0
            mask[-pad:, :, :] = 0
        else:
            mask[:, :pad, :] = 0
            mask[:, -pad:, :] = 0
    return mask

def fill_mask_up(mask, upperbound, lowerbound):
    """
    :param mask: H, W
    :param upperbound: the bound to fill up to, the lowerbound of upper mask
    :param lowerbound: the bound to start with, the lowerbound of current mask
    """
    upperleft, upperright = np.min(upperbound[:, 0]), np.max(upperbound[:, 0])
    lowerleft, lowerright = np.min(lowerbound[:, 0]), np.max(lowerbound[:, 0])
    upperleft, upperright = int(upperleft), int(upperright)
    lowerleft, lowerright = int(lowerleft), int(lowerright)

    mask_new = np.zeros_like(mask, dtype=mask.dtype)
    mask_new[:] = mask
    m = np.max(mask)

    H, W = mask.shape
    for w in range(W):
        if upperleft <= w <= upperright and lowerleft <= w <= lowerright:
            try:
                i = np.where(upperbound[:, 0] == w)
                j = np.where(lowerbound[:, 0] == w)
                i = int(upperbound[i[0][0], 1])
                j = int(lowerbound[j[0][0], 1])
                mask_new[i : j, w] = m
            except:
                pass

    return mask_new

def process_mask(mask):
    mask2 = np.zeros_like(mask)
    mask = mask.astype(np.float32)
    for i in range(3):
        img = mask[:, :, i]
        if np.sum(img) > 0:
            img = img_close(img, i)
            img = contour(img)
            # img = contour_convex(img)
        mask2[:, :, i] = img

    # fill holes vertically within each mask
    for i in range(1, 3):
        if np.sum(mask2[:, :, i]) > 0:
            upperbound = get_upperbound(mask2[:, :, i])
            lowerbound = get_lowerbound(mask2[:, :, i])
            newm = fill_mask_up(mask2[:, :, i], upperbound, lowerbound)
            mask2[:, :, i] = newm

    # fill the third part up to the lowerbound of second part
    if np.sum(mask2[:, :, 1]) > 0 and np.sum(mask2[:, :, 2]) > 0:
        upperbound = get_lowerbound(mask2[:, :, 1])
        lowerbound = get_lowerbound(mask2[:, :, 2])
        newm = fill_mask_up(mask2[:, :, 2], upperbound, lowerbound)
        mask2[:, :, 2] = newm

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


if __name__ == '__main__':
    cfg = Config().parse()

    image_dir = cfg.test_image_dir
    label_dir = cfg.test_label_dir if cfg.test_label_dir != 'None' else None
    pred_mask_dir = cfg.pred_mask_dir
    save_dir = cfg.plot_mask_dir
    W = cfg.W
    H = cfg.H
    topad = True if cfg.resize_with_pad == 'True' else False

    os.makedirs(save_dir, exist_ok=True)
    label_dict = get_label_dict(image_dir, label_dir)
    classes = cfg.classes
    if classes == 6:
        class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':3, 'PS BT':4, 'PS VS':5}
    elif classes == 3:
        class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':0, 'PS BT':1, 'PS VS':2}
    else:  # classes = 1
        class_index = {'STBD TS':0, 'STBD BT':0, 'STBD VS': 0, 'PS TS':0, 'PS BT':0, 'PS VS':0}
    # name = 'V3 13HR'

    # collect names with test result
    names = [os.path.splitext(f)[0] for f in os.listdir(pred_mask_dir) 
                                        if f.endswith('.npy')]

    for name in names:
        print('processing', name)
        label_info = label_dict[name]
        img = cv2.imread(label_info["path"])
        if topad:
            img, factor, direction, pad = resize_with_pad(img, W, H)
        else:
            img, w_factor, h_factor = resize_without_pad(img, W, H)
        if label_dir is not None:
            if topad:
                mask = make_mask_with_pad(label_info, class_index, factor, direction, pad, W, H)
            else:
                mask = make_mask_without_pad(label_info, class_index, w_factor, h_factor, W, H)
            mask = mask.astype(np.uint8)
            # print(img.shape, mask.shape)

        pred_mask = np.load(os.path.join(pred_mask_dir, name+'.npy'))
        pred_mask = pred_mask > 0.5  # convert to binary mask
        pred_mask = pred_mask.astype(np.uint8)
        if len(set(class_index.values())) == 6:
            pred_mask = bin_mask(pred_mask)
        # print(pred_mask.shape)

        # # save img and mask
        # cv2.imwrite(os.path.join(save_dir, name+'.jpg'), img)
        # if label_dir is not None:
        #     plot_mask(mask, os.path.join(save_dir, name+'_true.jpg'))
        # plot_mask(pred_mask, os.path.join(save_dir, name+'_pred.jpg'))

        # put mask on img
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if label_dir is not None:
            mask = get_3channelmask(mask)
            plot_mask_on_img(img, mask, os.path.join(save_dir, name+'_true.jpg'))
        pred_mask = get_3channelmask(pred_mask)
        if topad:
            pred_mask = zero_pad(pred_mask, factor, direction, pad)
        pred_mask = process_mask(pred_mask)
        plot_mask_on_img(img, pred_mask, os.path.join(save_dir, name+'_pred.jpg'))
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, name+'_pred_mask.jpg'), pred_mask)

        # break

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()