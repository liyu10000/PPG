import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed


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

def update_patch(manual_dir, manual_whole_dir, pred_dir, update_patch_dir, name, hws, cmd='union', patch_size=224):
    manual_label = cv2.imread(os.path.join(manual_dir, name+'.png'))
    manual_whole = cv2.imread(os.path.join(manual_whole_dir, name+'.png'))
    pred_label = cv2.imread(os.path.join(pred_dir, name+'.png'))
    for H, W, h, w in hws:
        manual_patch = manual_label[h:h+patch_size, w:w+patch_size]
        whole_patch = manual_whole[h:h+patch_size, w:w+patch_size]
        pred_patch = pred_label[h:h+patch_size, w:w+patch_size]
        if cmd == 'union':
            patch = np.where(manual_patch|pred_patch, 255, 0)
        elif cmd == 'intersect':
            patch = np.where(manual_patch&pred_patch, 255, 0)
        else: # use predicted label directly
            patch = pred_patch
        patch = np.where(whole_patch > 0, patch, 0)
        patch_name = '{}_H{}_W{}_h{}_w{}.png'.format(name, H, W, h, w)
        cv2.imwrite(os.path.join(update_patch_dir, patch_name), patch)

def update_patch_mp(manual_dir, manual_whole_dir, pred_dir, manual_patch_dir, update_patch_dir, cmd):
    file_dict = parse_patch_dir(manual_patch_dir)
    os.makedirs(update_patch_dir, exist_ok=True)
    print('# files', len(file_dict))

    executor = ProcessPoolExecutor(max_workers=4)
    tasks = []

    for name, hws in file_dict.items():
        tasks.append(executor.submit(update_patch, manual_dir, manual_whole_dir, pred_dir, update_patch_dir, name, hws, cmd))
    
    job_count = len(tasks)
    for future in as_completed(tasks):
        # res = future.result()
        job_count -= 1
        print("One Job Done, Remaining Job Count: %s" % (job_count))    


if __name__ == '__main__':
    epoch = '000'
    cat = 'df' # df, lowres-aug
    manual_dir = '../datadefects/mixquality/labels'
    manual_whole_dir = '../datadefects/mixquality/labels_whole'
    pred_dir = '../datadefects/exps/exp27/teacher_woLow_{}_joint'.format(epoch)
    manual_patch_dir = '../datadefects/mixquality-3cls-224/labels-{}'.format(cat)
    update_patch_dir = '../datadefects/exps/exp27/teacher_woLow_{}-{}'.format(epoch, cat)
    cmd = 'union'
    update_patch_mp(manual_dir, manual_whole_dir, pred_dir, manual_patch_dir, update_patch_dir, cmd)