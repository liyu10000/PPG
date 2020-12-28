import os
import sys
import cv2
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
from albumentations import RandomContrast, RandomBrightness, RandomBrightnessContrast, ToGray, JpegCompression
from albumentations.pytorch import ToTensor
from config import Config

cfg = Config().parse()
seed = cfg.seed
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)


def calc_mean_std(dataset):
    """ Calculate pixel level mean and std over all pixels of all images
    :param dataset: Dataset object, img and mask should be returned before
                    any augmentation, the shape of them is HxWxC (comment out
                    augment part in PPGDataset.__getitem__)
    """
    pixel_mean = np.zeros(3)
    pixel_std = np.zeros(3)
    k = 1
    for _, image, _ in tqdm(dataset, "Computing mean/std", len(dataset), unit="image"):
        image = np.array(image)
        pixels = image.reshape((-1, image.shape[2]))
        for pixel in pixels:
            diff = pixel - pixel_mean
            pixel_mean += diff / k
            pixel_std += diff * (pixel - pixel_mean)
            k += 1
    pixel_std = np.sqrt(pixel_std / (k - 2))
    pixel_mean /= 255.
    pixel_std /= 255.
    return pixel_mean, pixel_std


def get_transforms(phase):
    mean = (0.434, 0.447, 0.534)
    std = (0.281, 0.267, 0.271)
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5),
                # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                RandomContrast(p=0.5),
                RandomBrightness(p=0.5),
                # RandomBrightnessContrast(p=0.5),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


### Dataloader
class PPGDataset(Dataset):
    def __init__(self, pairs, phase, classes, useLow, weight):
        self.pairs = pairs
        self.phase = phase
        self.classes = classes
        self.useLow = useLow
        self.weight = weight
        self.transforms = get_transforms(phase)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        name = pair[0]
        img = cv2.imread(pair[1])
        mask = cv2.imread(pair[2], cv2.IMREAD_UNCHANGED)
        if not self.useLow: # remove low-q masks
            qua = cv2.imread(pair[3])
            mask[:,:,0] = np.where(qua[:,:,0], mask[:,:,0], 0)
            mask[:,:,1] = np.where(qua[:,:,0], mask[:,:,1], 0)
            mask[:,:,2] = np.where(qua[:,:,0], mask[:,:,2], 0)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image'] # CxHxW
        mask = augmented['mask'] # 1xHxWxC or 1xHxW
        # print(img.shape, mask.shape)
        mask = mask[0].permute(2, 0, 1) # CxHxW
        if self.classes == 1: # convert mask to grayscale
            mask = (mask.sum(dim=0, keepdim=True) > 0).type_as(mask)
        if self.weight != 1.0:
            weight = torch.where(mask > 0, torch.tensor(1.0), torch.tensor(self.weight))
        else:
            weight = torch.tensor(self.weight)
        return name, img, mask, weight

    def __len__(self):
        return len(self.pairs)


def generator(
            names,
            image_dir,
            label_dir,
            label_qua_dir,
            label_info_file,
            useLow,
            ratios,
            phase,
            classes,
            weight,
            train_val_split=[], # should be [partition numbers, partition index]
            batch_size=8,
            num_workers=4,
            ):
    if isinstance(image_dir, list):
        image_dir = image_dir[1:]
        label_dir = label_dir[1:]
        label_qua_dir = label_qua_dir[1:]
        label_info_file = label_info_file[1:]
    else:
        image_dir = [image_dir]
        label_dir = [label_dir]
        label_qua_dir = [label_qua_dir]
        label_info_file = [label_info_file]
    pairs = []
    names = set(names)
    for img_dir, lbl_dir, lbl_qua_dir, lbl_info_file in zip(image_dir, label_dir, label_qua_dir, label_info_file):
        keys = [f for f in os.listdir(img_dir) if (f.endswith('.png')) and (f.split('_H')[0] in names)]
        df = pd.read_csv(lbl_info_file)
        df.set_index('name', inplace=True)
        d = df.to_dict()
        newkeys = []
        for f in keys:
            if d['high_r'][f] > ratios[0] and d['low_r'][f] > ratios[1] and d['total_r'][f] > ratios[2]:
                newkeys.append(f)
        print('apply ratio threshold on patches: from {} to {}'.format(len(keys), len(newkeys)))
        pairs += [(f[:-4], os.path.join(img_dir, f), os.path.join(lbl_dir, f), os.path.join(lbl_qua_dir, f)) for f in newkeys]
    random.Random(seed).shuffle(pairs) # shuffle with seed, so that yielding same sampling
    if train_val_split:
        num, idx = train_val_split
        part_cnt = len(pairs) // num
        if phase == 'train':
            pairs = pairs[:part_cnt*idx] + pairs[part_cnt*(idx+1):]
        elif phase == 'val':
            pairs = pairs[part_cnt*idx : part_cnt*(idx+1)]
    print(phase, len(pairs))
    dataset = PPGDataset(pairs, phase, classes, useLow, weight)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(phase=='train'),   
    )
    return dataloader



if __name__ == "__main__":
    # test dataloader
    image_dir = "../datadefects/mixquality-3cls-224/images"
    label_dir = "../datadefects/mixquality-3cls-224/labels"
    label_qua_dir = '../datadefects/mixquality-3cls-224/labels_qua'
    label_info_file = '../datadefects/mixquality-3cls-224/labels_qua.csv'
    useLow = True
    ratios = [-1.0, 0.1, -1.0]
    phase = "val"
    classes = 1
    weight = 1.0
    train_val_split = [9, 0]

    if isinstance(image_dir, list):
        image_dir = image_dir[1:]
        label_dir = label_dir[1:]
        label_qua_dir = label_qua_dir[1:]
        label_info_file = label_info_file[1:]
    else:
        image_dir = [image_dir]
        label_dir = [label_dir]
        label_qua_dir = [label_qua_dir]
        label_info_file = [label_info_file]
    pairs = []
    for img_dir, lbl_dir, lbl_qua_dir, lbl_info_file in zip(image_dir, label_dir, label_qua_dir, label_info_file):
        keys = [f for f in os.listdir(img_dir) if (f.endswith('.png'))]
        df = pd.read_csv(lbl_info_file)
        newkeys = []
        for f in keys:
            row = df.loc[df['name'] == f]
            if row.iloc[0].high_r > ratios[0] and row.iloc[0].low_r > ratios[1] and row.iloc[0].total_r > ratios[2]:
                newkeys.append(f)
        print('apply ratio threshold on patches: from {} to {}'.format(len(keys), len(newkeys)))
        pairs += [(f[:-4], os.path.join(img_dir, f), os.path.join(lbl_dir, f), os.path.join(lbl_qua_dir, f)) for f in newkeys]
    random.Random(seed).shuffle(pairs) # shuffle with seed, so that yielding same sampling
    if train_val_split:
        num, idx = train_val_split
        part_cnt = len(pairs) // num
        if phase == 'train':
            pairs = pairs[:part_cnt*idx] + pairs[part_cnt*(idx+1):]
        elif phase == 'val':
            pairs = pairs[part_cnt*idx : part_cnt*(idx+1)]
    print(phase, len(pairs))
    dataset = PPGDataset(pairs, phase, classes, useLow, weight)

    tmp_dir = '../datadefects/mixquality-3cls-224/tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    for i in tqdm(range(len(dataset)), ncols=100):
        name, img, mask, weight = dataset[i]
        print(i, img.shape, mask.shape, weight.shape)
        mask = (mask.numpy()[0] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(tmp_dir, name+'.png'), mask)
        # print(mask.max(), mask.min())
        # print(weight.max(), weight.min())
        # break
