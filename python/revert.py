import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


def scan_files(directory, prefix=None, postfix=None):
    files_list = []
    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root, special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root, special_file))
            else:
                files_list.append(os.path.join(root, special_file))
    return files_list


def get_label_dict(image_dir, label_dir):
    images = os.listdir(image_dir)
    label_dict = {os.path.splitext(f)[0]:{'path':os.path.join(image_dir, f)} for f in images}
    if label_dir is None:
        return label_dict
    labels = scan_files(label_dir, postfix='.csv')
    has_labels = set() # store image names with labels
    for f in labels:
        base_f = os.path.basename(f)
        tokens = base_f.split()
        name = tokens[0] + ' ' + tokens[1].split('_')[0]
        side = tokens[1].split('_')[1]
        part = tokens[2]
        key = side + ' ' + part
        if not name in label_dict:  # no image
            continue
        if not key in label_dict[name]:
            label_dict[name][key] = []
        label_dict[name][key].append(f)
        has_labels.add(name)
    
    # remove keys without labels
    label_dict = {k:label_dict[k] for k in has_labels}

    # pprint(label_dict)
    return label_dict


def resize_with_pad(img, W, H):
    h, w, _ = img.shape  # usually we expect h >= H and w >= W
    factor = 1.0         # scaling factor, <= 1.0
    direction = "None"   # pad direction, can be None, Height, Width
    pad = 0
    if H / h == W / w:   # aspect ratio matches
        factor = W / w
        img = cv2.resize(img, (W, H))
    elif H / h > W / w:  # need to pad in height direction
        factor = W / w
        direction = "Height"
        h_ = int(h * factor)
        pad = int((H - h_) / 2)
        pad_ = H - h_ - pad
        img = cv2.resize(img, (W, h_))
        img = cv2.copyMakeBorder(img, pad, pad_, 0, 0, cv2.BORDER_CONSTANT, 0)  # pad with constant zeros
    else:                # need to pad in width direction
        factor = H / h
        direction = "Width"
        w_ = int(w * factor)
        pad = int((W - w_) / 2)
        pad_ = W - w_ - pad
        img = cv2.resize(img, (w_, H))
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad_, cv2.BORDER_CONSTANT, 0)  # pad with constant zeros
    return img, factor, direction, pad

def make_mask(label_info, class_index, factor, direction, pad, W, H):
    """ 
    :param class_index: {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':3, 'PS BT':4, 'PS VS':5}
    """
    class_num = len(set(class_index.values()))  # determine number of classes by class indices
    mask = np.zeros((class_num, H, W), dtype=np.float32)
    for side_part, fs in label_info.items():
        if side_part == "path":
            continue
        i = class_index[side_part]
        for f in fs:
            df = pd.read_csv(f)
            points = df.to_numpy(dtype=np.float32)
            points *= factor  # resize
            if pad > 0:
                if direction == "Height":
                    points[:, 1] += pad  # add pad to y
                else:
                    points[:, 0] += pad  # add pad to x
            cv2.fillConvexPoly(mask[i, :, :], points.astype(int), 1.0)  # mask[:, :, i] doesn't work
    mask = mask.transpose(1, 2, 0) # H, W, C
    return mask

def revert(image_dir, label_dir, image_src_dir, new_image_dir, new_label_dir):
    label_dict = get_label_dict(image_dir, label_dir)
    label_dict = {k:v for k,v in label_dict.items() if k.startswith('S')}
    years = {'5':2019, '6':2015, '9':2011, '11':2015, '14':2011, '19':2017, '20':2019}
    for name, label_info in label_dict.items():
        n1, n2 = name.split()
        n1 = n1[1:]
        n2 = n2[:-2]
        src_name = 'ship_{}_{}_image_{}_x2_SR.png'.format(n1, years[n1], n2)
        image_src_name = os.path.join(image_src_dir, src_name)
        print(os.path.isfile(image_src_name), image_src_name)
        img = cv2.imread(image_src_name)
        new_name = os.path.join(new_image_dir, name+'.png')
        cv2.imwrite(new_name, img)
        _, factor, direction, pad = resize_with_pad(img, 640, 480)
        for side_part, fs in label_info.items():
            if side_part == 'path':
                continue
            for f in fs:
                df = pd.read_csv(f)
                points = df.to_numpy(dtype=np.float32)
                if pad > 0:
                    if direction == 'Height':
                        points[:, 1] -= pad
                    else:
                        points[:, 0] -= pad
                points /= factor
                points = points.astype(int)
                points = np.where(points < 0, 0, points)
                df = pd.DataFrame(data={'X':points[:, 0], 'Y':points[:, 1]})
                new_f = os.path.join(new_label_dir, os.path.basename(f))
                df.to_csv(new_f, index=False)


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

def check_mask(image_dir, label_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    label_dict = get_label_dict(image_dir, label_dir)
    class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':0, 'PS BT':1, 'PS VS':2}
    names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.png')]

    for name in names:
        print('processing', name)
        label_info = label_dict[name]
        img = cv2.imread(label_info["path"])
        img, factor, direction, pad = resize_with_pad(img, 640, 480)
        if label_dir is not None:
            mask = make_mask(label_info, class_index, factor, direction, pad, 640, 480)
            mask = mask.astype(np.uint8)
            # print(img.shape, mask.shape)
        # put mask on img
        if label_dir is not None:
            mask = get_3channelmask(mask)
            plot_mask_on_img(img, mask, os.path.join(save_dir, name+'_mask.jpg'))


if __name__ == "__main__":
    # revert mask points
    image_dir = "../data/labeled/images"
    label_dir = "../data/labeled/labels"
    image_src_dir = "../data/Hi-Res-tmp"
    new_image_dir = "../data/labeled/images_new"
    new_label_dir = "../data/labeled/labels_new"
    os.makedirs(new_image_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)
    revert(image_dir, label_dir, image_src_dir, new_image_dir, new_label_dir)

    # put mask on img for checking
    image_dir = "../data/labeled/images_new"
    label_dir = "../data/labeled/labels_new"
    save_dir = "../data/labeled/masks"
    check_mask(image_dir, label_dir, save_dir)