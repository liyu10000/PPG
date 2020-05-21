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
    mean = (0.433, 0.445, 0.518)
    std = (0.277, 0.254, 0.266)
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
    def __init__(self, pairs, phase, classes, weight):
        self.pairs = pairs
        self.phase = phase
        self.classes = classes
        self.weight = weight # in the form [a, b, c]
        self.transforms = get_transforms(phase)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        name = pair[0]
        img = cv2.imread(pair[1])
        mask = cv2.imread(pair[2], cv2.IMREAD_UNCHANGED)
        if self.classes == 1 and len(mask.shape) == 3:
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # HxW
            gray = np.expand_dims(gray, axis=0) # 1xHxW
            augmented = self.transforms(image=img, mask=mask, gray=gray)
            gray = augmented['gray'] # 1xHxW
        else:
            augmented = self.transforms(image=img, mask=mask)
        img = augmented['image'] # CxHxW
        mask = augmented['mask'] # 1xHxWxC or 1xHxW
        # print(img.shape, mask.shape)
        if len(mask.shape) == 4: # when mask is non-grayscale
            mask = mask[0].permute(2, 0, 1) # CxHxW
        if self.weight and mask.shape[0] == 1:
            print('Cannot assign weight when raw input mask is grayscale')
            sys.exit()
        C, H, W = mask.shape
        weight = torch.tensor(self.weight)
        if self.weight:
            if self.classes == 1:
                weight = torch.ones_like(mask)
                for i in range(C):
                    weight[i, :, :] += mask[i, :, :]* (self.weight[i] - 1)
                weight = weight.max(dim=0, keepdim=True).values
            else: # classes = 3
                weight = weight.view(C, 1).expand(C, H*W).view(C, H, W)
        if self.classes == 1:
            return name, img, gray, weight
        else:
            return name, img, mask, weight

    def __len__(self):
        return len(self.pairs)


def generator(
            image_dir,
            label_dir,
            phase,
            classes,
            weight,
            batch_size=8,
            num_workers=4,
            ):
    if isinstance(image_dir, list):
        image_dir = image_dir[1:]
        label_dir = label_dir[1:]
    else:
        image_dir = [image_dir]
        label_dir = [label_dir]
    pairs = []
    for img_dir, lbl_dir in zip(image_dir, label_dir):
        keys = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        pairs += [(f[:-4], os.path.join(img_dir, f), os.path.join(lbl_dir, f)) for f in keys]
    random.Random(seed).shuffle(pairs) # shuffle with seed, so that yielding same sampling
    print(phase, len(pairs))
    dataset = PPGDataset(pairs, phase, classes, weight)
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
    image_dir = "../datadefects/highquality-3cls-224/images5"
    label_dir = "../datadefects/highquality-3cls-224/labels5"
    phase = "val"
    classes = 1
    weight = [1.0, 2.0, 3.0]
    if isinstance(image_dir, list):
        image_dir = image_dir[1:]
        label_dir = label_dir[1:]
    else:
        image_dir = [image_dir]
        label_dir = [label_dir]
    pairs = []
    for img_dir, lbl_dir in zip(image_dir, label_dir):
        keys = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        pairs += [(f[:-4], os.path.join(img_dir, f), os.path.join(lbl_dir, f)) for f in keys]
    dataset = PPGDataset(pairs, phase, classes, weight)

    for i in range(len(dataset)):
        print(i)
        name, img, mask, weight = dataset[i]
        print(img.shape, mask.shape, weight.shape)
        print(mask.max(), mask.min())
        print(weight.max(), weight.min())
        break
