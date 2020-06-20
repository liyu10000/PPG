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
    
def show_labels(f, file_dict, save_dir=None):
    d = file_dict[f]
    path = d['path']
    # get image path and read in
    img_path = os.path.join(path, d['img'])
    img = cv2.imread(img_path)
    # create mask
    mask = np.zeros_like(img, dtype=img.dtype)
    # get label path and plot it on image
    defects = ['corrosion', 'delamination', 'fouling']
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for defect,color in zip(defects,colors):
        csvs = d[defect]
        csvs = [os.path.join(path, csv) for csv in csvs]
        labels = read_labels(csvs)
        plot_labels(mask, labels, color, fill=True)
    if save_dir is None:
        f = plt.figure(1, figsize=(20,15))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(mask)
        plt.show()
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, f+'.png')
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_name, mask)

def slice(data_dir, save_dir, step_size, patch_size, binary=True, whole_mask_dir=None):
    """ Slice image&mask pairs into small patches
    @params
        data_dir: directory with two folders: images, labels
        save_dir: directory to save patches of images and labels
        step_size: step size at slicing
        patch_size: patch size
        binary: whether or not to save masks as binary or RGB
        whole_mask_dir: directory with whole vessel segmentation masks
    """
    image_dir = os.path.join(data_dir, 'images3')
    label_dir = os.path.join(data_dir, 'labels3')
    image_dir2 = os.path.join(save_dir, 'images3')
    label_dir2 = os.path.join(save_dir, 'labels3')
    os.makedirs(image_dir2, exist_ok=True)
    os.makedirs(label_dir2, exist_ok=True)
    files = os.listdir(image_dir)
    assert len(files) == len(os.listdir(label_dir))
    for f in files:
        basename = os.path.splitext(f)[0]
        print('Processing', basename)
        image_path = os.path.join(image_dir, f)
        label_path = os.path.join(label_dir, f)
        if whole_mask_dir is not None:
            whole_path = os.path.join(whole_mask_dir, f)
            whole = cv2.imread(whole_path)
        img = cv2.imread(image_path)
        mask = cv2.imread(label_path)
        H, W, _ = img.shape
        for h in range(0, H, step_size):
            for w in range(0, W, step_size):
                if h+patch_size > H:
                    h = H - patch_size
                if w+patch_size > W:
                    w = W - patch_size
                img_patch = img[h:h+patch_size, w:w+patch_size]
                mask_patch = mask[h:h+patch_size, w:w+patch_size]

                # # data augmentation on delamination
                # if np.any(mask_patch[:, :, 1] > 0):
                #     # randomly shift cut positions
                #     h += random.randint(-step_size, step_size)
                #     w += random.randint(-step_size, step_size)
                #     # ensure patch is available
                #     h = max(0, min(H - patch_size, h))
                #     w = max(0, min(W - patch_size, w))
                #     # recut patch
                #     img_patch = img[h:h+patch_size, w:w+patch_size]
                #     mask_patch = mask[h:h+patch_size, w:w+patch_size]
                #     # if new patch does not include delamination, skip
                #     if np.all(mask_patch[:, :, 1] == 0):
                #         continue
                # else:
                #     continue

                # check if patch is within whole vessel mask
                if whole_mask_dir is not None:
                    whole_patch = whole[h:h+patch_size, w:w+patch_size]
                    if np.sum(whole_patch) == 0: # ignore background patches
                        continue
                image_path2 = os.path.join(image_dir2, '{}_H{}_W{}_h{}_w{}.png'.format(basename, H, W, h, w))
                label_path2 = os.path.join(label_dir2, '{}_H{}_W{}_h{}_w{}.png'.format(basename, H, W, h, w))
                cv2.imwrite(image_path2, img_patch)
                if binary: # convert to binary mask
                    mask_patch = np.sum(mask_patch, axis=2)
                cv2.imwrite(label_path2, mask_patch)


if __name__ == '__main__':
    # # read csv files and generate masks
    # data_dir = '../datadefects/raw/lowquality2'
    # file_dict = parse_dir(data_dir)
    # save_dir = '../datadefects/lowquality/labels2'
    # for f in file_dict:
    #     print('Processing', f)
    #     show_labels(f, file_dict, save_dir)

    # slice images/labels into small patches
    patch_size = 224
    step_size = patch_size // 2
    data_dir = '../datadefects/lowquality'
    save_dir = '../datadefects/lowquality-3cls-224'
    binary = False
    whole_mask_dir = '../datadefects/labels_whole/set3'
    slice(data_dir, save_dir, step_size, patch_size, binary, whole_mask_dir)
