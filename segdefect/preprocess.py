import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_dir(data_dir):
    file_dirs = os.listdir(data_dir)
    file_dict = {f:{} for f in file_dirs}
    defects = ['corrosion', 'delamination', 'fouling']
    for f in file_dirs:
        path = os.path.join(data_dir, f)
        file_dict[f]['path'] = path
        file_dict[f]['img'] = f + '.png'
        for defect in defects:
            file_dict[f][defect] = []
        csvs = [c for c in os.listdir(path) if c.endswith('.csv')]
        csvs.sort()
        for csv in csvs:
            for defect in defects:
                if defect in csv:
                    file_dict[f][defect].append(csv)
    return file_dict

def read_labels(csvs):
    labels = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
            xy = df.to_numpy()
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

def show_labels_on_image(f, file_dict):
    d = file_dict[f]
    path = d['path']
    # get image path and read in
    img_path = os.path.join(path, d['img'])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get label path and plot it on image
    defects = ['corrosion', 'delamination', 'fouling']
    colors = [(255,0,0), (255,255,0), (0,255,0)]
    for defect,color in zip(defects,colors):
        csvs = d[defect]
        csvs = [os.path.join(path, csv) for csv in csvs]
        labels = read_labels(csvs)
        plot_labels(img, labels, color)
    f = plt.figure(1, figsize=(20,15))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()
    
def show_labels(f, file_dict, save_dir, wh=None):
    d = file_dict[f]
    path = d['path']
    # get image path and read in
    img_path = os.path.join(path, d['img'])
    img = cv2.imread(img_path)
    if wh is not None:
        H, W, _ = img.shape
        w, h = wh
        scale = np.array([w / W, h / H]).reshape(1, 2)
        img = cv2.resize(img, (w, h))
    # create mask
    mask = np.zeros_like(img, dtype=img.dtype)
    # get label path and plot it on image
    defects = ['corrosion', 'delamination', 'fouling']
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for defect,color in zip(defects,colors):
        csvs = d[defect]
        csvs = [os.path.join(path, csv) for csv in csvs]
        labels = read_labels(csvs)
        if wh is not None:
            labels = [np.multiply(label, scale) for label in labels]
        labels = [np.array(label).astype(int) for label in labels]
        plot_labels(mask, labels, color, fill=True)

    save_name = os.path.join(save_dir, f+'.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_name, mask)

def csv_to_masks(data_dir, save_dir, wh=None):
    os.makedirs(save_dir, exist_ok=True)
    file_dict = parse_dir(data_dir)
    for f in file_dict:
        print('Processing', f)
        show_labels(f, file_dict, save_dir, wh)

def gen_defect_data(image_dir, label_dir, 
                    patch_image_dir, patch_label_dir, 
                    step_size, patch_size, 
                    defect_only=True,
                    binary=True, 
                    whole_mask_dir=None):
    """ Slice image&mask pairs into small patches
    @params
        image_dir, label_dir: directory with two folders: images, labels
        patch_image_dir, patch_label_dir: directory to save patches of images and labels
        step_size: step size at slicing
        patch_size: patch size
        defect_only: if to only keep patches with defects
        binary: whether or not to save masks as binary or RGB
        whole_mask_dir: directory with whole vessel segmentation masks
    """
    os.makedirs(patch_image_dir, exist_ok=True)
    if label_dir:
        os.makedirs(patch_label_dir, exist_ok=True)
    # defect_only = False if not label_dir else defect_only
    files = os.listdir(image_dir)
    # assert len(files) == len(os.listdir(label_dir))
    for f in files:
        basename = os.path.splitext(f)[0]
        print('Processing', basename)
        image_path = os.path.join(image_dir, f)
        img = cv2.imread(image_path)
        H, W, _ = img.shape
        if label_dir:
            label_path = os.path.join(label_dir, f)
            mask = cv2.imread(label_path)
        if whole_mask_dir:
            whole_path = os.path.join(whole_mask_dir, f)
            whole = cv2.imread(whole_path)
        
        for h in range(0, H, step_size):
            for w in range(0, W, step_size):
                if h+patch_size > H:
                    h = H - patch_size
                if w+patch_size > W:
                    w = W - patch_size
                img_patch = img[h:h+patch_size, w:w+patch_size]

                if label_dir:
                    mask_patch = mask[h:h+patch_size, w:w+patch_size]

                    # keep those with defects
                    if defect_only and np.all(mask_patch == 0):
                        continue

                    # # data augmentation
                    # if np.any(mask_patch > 0):
                    #     # randomly shift cut positions
                    #     h += random.randint(-step_size, step_size)
                    #     w += random.randint(-step_size, step_size)
                    #     # ensure patch is available
                    #     h = max(0, min(H - patch_size, h))
                    #     w = max(0, min(W - patch_size, w))
                    #     # recut patch
                    #     img_patch = img[h:h+patch_size, w:w+patch_size]
                    #     mask_patch = mask[h:h+patch_size, w:w+patch_size]
                    #     # if new patch does not include any defects, skip
                    #     if np.all(mask_patch == 0):
                    #         continue
                    # else:
                    #     continue

                # check if patch is within whole vessel mask
                if whole_mask_dir:
                    whole_patch = whole[h:h+patch_size, w:w+patch_size]
                    if np.sum(whole_patch) == 0: # ignore background patches
                        continue
                        
                patch_image_path = os.path.join(patch_image_dir, '{}_H{}_W{}_h{}_w{}.png'.format(basename, H, W, h, w))
                cv2.imwrite(patch_image_path, img_patch)
                if label_dir:
                    patch_label_path = os.path.join(patch_label_dir, '{}_H{}_W{}_h{}_w{}.png'.format(basename, H, W, h, w))
                    if binary: # convert to binary mask
                        mask_patch = np.sum(mask_patch, axis=2)
                    cv2.imwrite(patch_label_path, mask_patch)

