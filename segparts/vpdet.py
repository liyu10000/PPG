#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def organize_files(results_dir):
    names = [os.path.splitext(f)[0] for f in os.listdir(results_dir) if f.endswith('.jpg') or f.endswith('.png')]
    files = {name:[] for name in names}
    ptsfiles = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    for f in ptsfiles:
        for name in names:
            if f.startswith(name):
                files[name].append(os.path.join(results_dir, f))
                break
    return files

def read_lines(ptsfiles):
    xys = []
    for f in ptsfiles:
        df = pd.read_csv(f, header=None)
        xy = df.to_numpy()
        xy = xy.transpose((1, 0))
        # reverse points to make span from left to right
        if xy[0, 0] > xy[-1, 0]:
            xy = np.flip(xy, 0)
        xys.append(xy)
    return xys

def keep_horizontals(xys):
    new_xys = []
    for xy in xys:
        if abs(xy[0, 0] - xy[-1, 0]) < 1e-4: # remove vertical lines
            continue
        c = np.polyfit(xy[:, 0], xy[:, 1], 1)
        if -2 < c[0] < 2:
            new_xys.append(xy)
    return new_xys

def calc_angle(a1, a2):
    angle = (a1 - a2) / (1 + a1 * a2)
    angle = np.arctan(angle) / math.pi * 180
    return angle
    
def isinline(xy1, xy2, c1, c2, maxangle=3):
    min1, min2 = xy1[0, 0], xy2[0, 0]
    if min2 < min1:
        xy1, xy2 = xy2, xy1
        c1, c2 = c2, c1
    min1, max1 = xy1[0, 0], xy1[-1, 0]
    min2, max2 = xy2[0, 0], xy2[-1, 0]
    span1, span2 = max1-min1, max2-min2 # span in horizontal
    len1, len2 = xy1.shape[0], xy2.shape[0] # number of points
    if max1 < min2:
        spanc = min2 - max1
        if spanc > span1 or spanc > span2: # do not consider two segments far away
            return False, None
#         connect = np.array([xy1[-1,:], xy2[0,:]]) # connect two end points in the middle
        connect = np.array([xy1[-len1//10,:], xy2[len2//10,:]]) # connect two end points in the middle
#         connect = np.concatenate([xy1, xy2], axis=0)
        cc = np.polyfit(connect[:, 0], connect[:, 1], 1)
        inline = (abs(calc_angle(c1[0], c2[0])) < maxangle) and \
                 (abs(calc_angle(c1[0], cc[0])) < maxangle) and \
                 (abs(calc_angle(c2[0], cc[0])) < maxangle)
        xy = np.concatenate([xy1, xy2], axis=0)
        return inline, xy
    else:
        if abs(calc_angle(c1[0], c2[0])) < maxangle and abs(c1[1]-c2[1])<10: # merge two overlapping segments
            connect = np.array([xy1[-len1//10,:], xy2[len2//10,:]])
#             connect = np.concatenate([xy1, xy2], axis=0)
            cc = np.polyfit(connect[:, 0], connect[:, 1], 1)
            inline = (abs(calc_angle(c1[0], c2[0])) < maxangle) and \
                     (abs(calc_angle(c1[0], cc[0])) < maxangle) and \
                     (abs(calc_angle(c2[0], cc[0])) < maxangle)
            if inline:
                xy = np.concatenate([xy1, xy2], axis=0)
                cc = np.polyfit(xy[:, 0], xy[:, 1], 1)
                minm, maxm = min(min1, min2), max(max1, max2)
                xy = np.array([[minm, cc[0]*minm+cc[1]], [maxm, cc[0]*maxm+cc[1]]])
                return True, xy
        return False, None
    
def merge_lines(xys):
    # sort lines by starting point
    xys = [xy for xy in sorted(xys, key=lambda xy:xy[0,0])]
    # merge two lines if they have close slopes and don't overlap
    new_xys = []
    while len(xys) > 1:
        xy1 = xys.pop(0)
        merged = False
        c1 = np.polyfit(xy1[:, 0], xy1[:, 1], 1)
        for i,xy2 in enumerate(xys):
            c2 = np.polyfit(xy2[:, 0], xy2[:, 1], 1)
            inline, xy = isinline(xy1, xy2, c1, c2)
            if inline:
                xys.pop(i)
                xys.append(xy)
                merged = True
                break
        if not merged:
            new_xys.append(xy1)
    new_xys.extend(xys)
    return new_xys

def remove_short(xys, W=640, H=480, factor=4):
    new_xys = []
    for xy in xys:
        hspan = xy[-1, 0] - xy[0, 0]
        vspan = xy[-1, 1] - xy[0, 1]
        if hspan > W / factor or vspan > H / factor:
            new_xys.append(xy)
    return new_xys

def four_longest(xys, hull=None):
    # remove segments outside of hull
    new_xys = []
    for xy in xys:
        left, middle, right = xy[0, :], np.mean(xy, axis=0), xy[-1, :]
        count = 0
        count += 1 if cv2.pointPolygonTest(hull, tuple(left), False) >= 0 else 0
        count += 1 if cv2.pointPolygonTest(hull, tuple(middle), False) >= 0 else 0
        count += 1 if cv2.pointPolygonTest(hull, tuple(right), False) >= 0 else 0
        if count >= 2:
            new_xys.append(xy)
    xys = new_xys
    # sort by length of segments
    lengths = [] 
    for xy in xys:
        hspan = abs(xy[-1, 0] - xy[0, 0])
        vspan = abs(xy[-1, 1] - xy[0, 1])
        lengths.append(hspan**2 + vspan**2)
    xys = [xy for _,xy in sorted(zip(lengths, xys), key=lambda pair:pair[0], reverse=True)]
    # take the first four segments and sort them by height
    xys = xys[:4]
    xys = [xy for xy in sorted(xys, key=lambda xy:np.mean(xy[:, 1]))]
    return xys


if __name__ == '__main__':
    img_dir = './data'
    results_dir = './results'
    files = organize_files(results_dir)


    fig = plt.figure(1, figsize=(10, 22))
    names = list(files.keys())
    names.sort()
    names.remove('11018')
    for i,name in enumerate(names):
        ax = fig.add_subplot(5, 2, i+1)
        img = cv2.imread(os.path.join(img_dir, name+'.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        
        ptsfiles = files[name]
        xys = read_lines(ptsfiles)
        xys = keep_horizontals(xys)
        xys = merge_lines(xys)
        xys = remove_short(xys)
        for i,xy in enumerate(xys):
            x, y = xy[:, 0], xy[:, 1]
            ax.plot(x, y, linewidth=4, color='red')
            ax.text(x[0], y[0], str(i))
        
        plt.xticks([])
        plt.yticks([])
    plt.show()
