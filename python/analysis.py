import os
import cv2
import numpy as np
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


if __name__ == "__main__":
    data_dir = "../data/labeled"
    seed_images = scan_files(data_dir, postfix="HR.png")
    seed_images.sort()
    print("# of images:", len(seed_images))
    
    # get_size(seed_images)
    get_label(data_dir)