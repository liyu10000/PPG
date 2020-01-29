import os
import cv2
import numpy as np
import pandas as pd
from pprint import pprint


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


def get_size(image_names):
    for image_name in image_names:
        img = cv2.imread(image_name)
        print(image_name, img.shape, img.shape[1] / img.shape[0])


def get_label(data_dir):
    label_csvs = scan_files(data_dir, postfix=".csv")
    summ = {"PS": {"TS":[0]*5, "BT":[0]*5, "VS":[0]*5},
            "STBD": {"TS":[0]*5, "BT":[0]*5, "VS":[0]*5}}
    for f in label_csvs:
        f = os.path.basename(f)
        tokens = f.split()
        side = tokens[1].split('_')[1]       # PS or STBD
        part = tokens[2]                     # TS, BT, or VS
        pnum = int(tokens[3].split('.')[1])  # 1, 2, 3, or 4
        summ[side][part][0] += 1
        summ[side][part][pnum] += 1
    print("# of csvs:", len(label_csvs))
    pprint(summ)


def resize_with_pad(img, W=640, H=480):
    h, w, _ = img.shape
    factor = W / w  # scaling factor, <= 1.0
    if H / h == factor:  # aspect ratio matches
        pad = 0
        img = cv2.resize(img, (W, H))
    else:  # need to pad in height direction
        h_ = int(h * factor)
        pad = int((H - h_) / 2)
        img = cv2.resize(img, (W, h_))
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, 0)  # pad with constant zeros
    return img, factor, pad


def get_label_dict(image_dir, label_dir):
    images = os.listdir(image_dir)
    labels = scan_files(label_dir, postfix='.csv')
    label_dict = {os.path.splitext(f)[0]:{} for f in images}
    for f in labels:
        base_f = os.path.basename(f)
        tokens = base_f.split()
        name = tokens[0] + ' ' + tokens[1].split('_')[0]
        side = tokens[1].split('_')[1]
        part = tokens[2]
        key = side + ' ' + part
        if not key in label_dict[name]:
            label_dict[name][key] = []
        label_dict[name][key].append(f)

    # pprint(label_dict)
    return label_dict


def make_mask(name, label_dict, class_index, factor, pad, W=640, H=480):
    """ 
    :param name: image name, like 'V9 50HR'
    :param class_index: {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':3, 'PS BT':4, 'PS VS':5}
    """
    label_info = label_dict[name]
    class_num = len(class_index)
    mask = np.zeros((class_num, H, W), dtype=np.float32)
    for side_part, fs in label_info.items():
        i = class_index[side_part]
        for f in fs:
            df = pd.read_csv(f)
            points = df.to_numpy(dtype=np.float32)
            points *= factor  # resize
            if pad > 0:
                points[:, 1] += pad  # add pad to y
            cv2.fillConvexPoly(mask[i, :, :], points.astype(int), 1.0)  # mask[:, :, i] doesn't work
    mask = mask.transpose(1, 2, 0) # H, W, C
    return mask



if __name__ == "__main__":
    data_dir = "../data/labeled"
    seed_images = scan_files(data_dir, postfix="HR.png")
    seed_images.sort()
    print("# of images:", len(seed_images))
    
    # get_size(seed_images)
    # get_label(data_dir)

    image_dir = "../data/labeled/images"
    label_dir = "../data/labeled/labels"
    label_dict = get_label_dict(image_dir, label_dir)
    class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':3, 'PS BT':4, 'PS VS':5}
    name = 'V4 23HR'

    img = cv2.imread(os.path.join(image_dir, name+'.png'))
    img, factor, pad = resize_with_pad(img)
    mask = make_mask(name, label_dict, class_index, factor, pad)
    print(img.shape, mask.shape)