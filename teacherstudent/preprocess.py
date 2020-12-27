import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_dir(data_dir):
    file_dirs = os.listdir(data_dir)
    file_dict = {f:{} for f in file_dirs}
    qualities = ['high', 'low']
    for f in file_dirs:
        path = os.path.join(data_dir, f)
        file_dict[f]['path'] = path
        file_dict[f]['img'] = f + '.png'
        for quality in qualities:
            file_dict[f][quality] = []
        csvs = [c for c in os.listdir(path) if c.endswith('.csv')]
        csvs.sort()
        for csv in csvs:
            if csv.endswith('high.csv'):
                file_dict[f]['high'].append(csv)
            else:
                file_dict[f]['low'].append(csv)
    return file_dict

def read_labels(csvs):
    labels = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
            xy = df.to_numpy().astype(int)
            labels.append(xy)
        except:
            print('corrupted: ', csv)
    return labels

def plot_labels(img, labels, color, fill=False):
    for label in labels:
        if fill: # filled polygon
            cv2.fillPoly(img, [label], color)
        else:
            cv2.polylines(img, [label], True, color)
    return img
    
def show_labels(f, file_dict, save_dir, wh=None):
    d = file_dict[f]
    path = d['path']
    # get image path and read in
    img_path = os.path.join(path, d['img'])
    img = cv2.imread(img_path)
    H, W, C = img.shape
    if wh is not None:
        w, h = wh
        scale = np.array([w / W, h / H]).reshape(1, 2)
        img = cv2.resize(img, (w, h))
        H, W = h, w
    # create mask
    mask = np.zeros((H, W, C), dtype=img.dtype)
    # get label path and plot it on image
    qualities = ['high', 'low']
    # colors = [(255,0,0), (0,255,0)] # ERROR here!!!
    for i, quality in enumerate(qualities):
        csvs = d[quality]
        csvs = [os.path.join(path, csv) for csv in csvs]
        labels = read_labels(csvs)
        if wh is not None:
            labels = [np.multiply(label, scale).astype(int) for label in labels]
        mask[:,:,i] = plot_labels(np.zeros((H, W), dtype=mask.dtype), labels, 255, fill=True)
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, f+'.png')
    cv2.imwrite(save_name, mask)

def parse_patch_dir(patch_dir):
    files = os.listdir(patch_dir)
    files.sort()
    file_dict = defaultdict(list)
    for f in files:
        tokens = f[:-4].rsplit('_', 4)
        name = tokens[0]
        H = int(tokens[1][1:])
        W = int(tokens[2][1:])
        h = int(tokens[3][1:])
        w = int(tokens[4][1:])
        file_dict[name].append((H, W, h, w))
    return file_dict

def slice_quality_mask(quality_dir, save_dir, name, hws, patch_size=224):
    img_name = os.path.join(quality_dir, name+'.png')
    img = cv2.imread(img_name)
    patch_names = []
    ratios = []
    area = patch_size ** 2
    for H, W, h, w in hws:
        patch = img[h:h+patch_size, w:w+patch_size]
        patch_name = '{}_H{}_W{}_h{}_w{}.png'.format(name, H, W, h, w)
        patch_names.append(patch_name)
        cv2.imwrite(os.path.join(save_dir, patch_name), patch)
        high_r = np.sum(patch[:,:,0]) / 255 / area
        low_r  = np.sum(patch[:,:,1]) / 255 / area
        patch  = np.where(patch[:,:,0]|patch[:,:,1], 1, 0)
        total_r = np.sum(patch) / area
        ratios.append([high_r, low_r, total_r])
    return patch_names, ratios 

def slice_quality_mask_batch(patch_dir, quality_dir, save_dir, patch_info_csv):
    file_dict = parse_patch_dir(patch_dir)
    os.makedirs(save_dir, exist_ok=True)
    print('# files', len(file_dict))

    executor = ProcessPoolExecutor(max_workers=4)
    tasks = []

    for name, hws in file_dict.items():
        tasks.append(executor.submit(slice_quality_mask, quality_dir, save_dir, name, hws))
    
    Names, Ratios = [], []
    job_count = len(tasks)
    for future in as_completed(tasks):
        patch_names, ratios = future.result()
        Names += patch_names
        Ratios += ratios
        job_count -= 1
        print("One Job Done, Remaining Job Count: %s" % (job_count))    

    # save ratio info into csv
    Ratios = np.array(Ratios)
    data = {'name':Names, 'high_r':Ratios[:, 0], 'low_r':Ratios[:, 1], 'total_r':Ratios[:, 2]}
    df = pd.DataFrame(data)
    df.to_csv(patch_info_csv, index=False)


if __name__ == '__main__':
    # # read csv files and generate high/low quality masks
    # data_dir = '../datadefects/raw/mixquality'
    # file_dict = parse_dir(data_dir)
    # save_dir = '../datadefects/mixquality/labels_qua'
    # wh = None # (640, 480)
    # for f in file_dict:
    #     print('Processing', f)
    #     show_labels(f, file_dict, save_dir, wh)

    # generate quality mask patches and calculate ratios
    patch_dir = '../datadefects/mixquality-3cls-224/labels'
    quality_dir = '../datadefects/mixquality/labels_qua'
    save_dir = '../datadefects/mixquality-3cls-224/labels_qua'
    patch_info_csv = '../datadefects/mixquality-3cls-224/labels_qua.csv'
    slice_quality_mask_batch(patch_dir, quality_dir, save_dir, patch_info_csv)