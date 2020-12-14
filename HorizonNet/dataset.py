import os
import cv2
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from utils import read_txt


class PPGPartDataset(Dataset):
    def __init__(self, img_txt_pairs, phase, return_bon=True, return_path=False):
        self.img_txt_pairs = img_txt_pairs
        self.phase = phase
        self.return_bon = return_bon
        self.return_path = return_path

    def __len__(self):
        return len(self.img_txt_pairs)

    def __getitem__(self, idx):
        # Read image
        img = cv2.imread(self.img_txt_pairs[idx][0]) / 255.
        H, W = img.shape[:2]

        if self.return_bon:
            # Read ground truth boundaries
            txt_content = read_txt(self.img_txt_pairs[idx][1])
            w1_begin, w1_end = txt_content['w1_begin'], txt_content['w1_end']
            w2_begin, w2_end = txt_content['w2_begin'], txt_content['w2_end']
            bon = txt_content['bon'] # wx2
            bon = np.array(bon, dtype=np.float32).transpose((1, 0)) # 2xw
            bon = bon / H * 4# scale distances
            bon_mask = np.zeros((2, W), dtype=np.float32)
            bon_mask[0][w1_begin:w1_end+1] = 1
            bon_mask[1][w2_begin:w2_end+1] = 1

        # Random flip
        if self.phase == 'train' and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            if self.return_bon:
                bon = np.flip(bon, axis=1)
                bon_mask = np.flip(bon_mask, axis=1)

        # Random gamma augmentation
        if self.phase == 'train' and np.random.randint(2) == 0:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img ** p

        # Convert to tensor and Check whether additional output are requested
        out_lst = [torch.FloatTensor(img.transpose([2, 0, 1]).copy())]
        if self.return_bon:
            out_lst.append(torch.FloatTensor(bon.copy()))
            out_lst.append(torch.FloatTensor(bon_mask.copy()))
        if self.return_path:
            out_lst.append(self.img_txt_pairs[idx][0])

        return out_lst


def generator(names,
              img_dirs, 
              txt_dirs, 
              phase,
              seed=42,
              train_val_split='', # should be 'partition numbers,partition index'
              return_bon=True, 
              return_path=False,
              batch_size=4,
              num_workers=4):
    img_txt_pairs = []
    for i, img_dir in enumerate(img_dirs):
        fs = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.png')]
        keys = []
        for f in fs:
            if f.startswith('ship'):
                if len(f.split('_', 7)) == 7: # original data, no aug
                    if f in names:
                        keys.append(f)
                else:
                    if '_'.join(f.split('_', 7)[:7]) in names:
                        keys.append(f)
            else:
                if len(f.split('_', 1)) == 1: # original data, no aug
                    if f in names:
                        keys.append(f)
                else:
                    if f.split('_', 1)[0] in names:
                        keys.append(f)
        keys.sort()
        if return_bon:
            txt_dir = txt_dirs[i]
            img_txt_pairs += [(os.path.join(img_dir, f+'.png'), os.path.join(txt_dir, f+'.txt')) for f in keys]
        else:
            img_txt_pairs += [(os.path.join(img_dir, f+'.png'), ) for f in keys]
    random.Random(seed).shuffle(img_txt_pairs) # shuffle with seed, so that yielding same sampling
    if train_val_split:
        num, idx = train_val_split.split(',')
        num, idx = int(num), int(idx)
        part_cnt = len(img_txt_pairs) // num
        if phase == 'train':
            img_txt_pairs = img_txt_pairs[:part_cnt*idx] + img_txt_pairs[part_cnt*(idx+1):]
        elif phase == 'valid':
            img_txt_pairs = img_txt_pairs[part_cnt*idx : part_cnt*(idx+1)]
    print(phase, len(img_txt_pairs))
    dataset = PPGPartDataset(img_txt_pairs, phase, return_bon, return_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(phase=='train'),   
    )
    return dataloader