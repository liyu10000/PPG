import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
from albumentations import RandomContrast, RandomBrightness, RandomBrightnessContrast, ToGray, JpegCompression
from albumentations.pytorch import ToTensor


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
    def __init__(self, pairs, phase, classes):
        self.pairs = pairs
        self.phase = phase
        self.classes = classes
        self.transforms = get_transforms(phase)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        name = pair[0]
        img = cv2.imread(pair[1])
        if len(pair) > 2:
            mask = cv2.imread(pair[2], cv2.IMREAD_UNCHANGED)
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image'] # CxHxW
            mask = augmented['mask'] # 1xHxWxC or 1xHxW
            # print(img.shape, mask.shape)
            if len(mask.shape) == 4: # when mask is non-grayscale
                mask = mask[0].permute(2, 0, 1) # CxHxW
            C, H, W = mask.shape
            if self.classes == 1 and C == 3: # convert mask to grayscale
                mask = (mask.sum(dim=0, keepdim=True) > 0).type_as(mask)
            return name, img, mask
        else:
            augmented = self.transforms(image=img)
            img = augmented['image'] # CxHxW
            return name, img, 0

    def __len__(self):
        return len(self.pairs)


def generator(
            image_dir,
            label_dir,
            phase,
            classes,
            train_val_split=[], # should be [partition numbers, partition index]
            batch_size=8,
            num_workers=4,
            ):
    if not isinstance(image_dir, list):
        image_dir = [image_dir]
        label_dir = [label_dir] if label_dir else []
    pairs = []
    if label_dir: # comes with ground truth labels
        for img_dir, lbl_dir in zip(image_dir, label_dir):
            keys = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            pairs += [(f[:-4], os.path.join(img_dir, f), os.path.join(lbl_dir, f)) for f in keys]
    else:
        for img_dir in image_dir:
            keys = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            pairs += [(f[:-4], os.path.join(img_dir, f)) for f in keys]
    # random.Random(seed).shuffle(pairs) # shuffle with seed, so that yielding same sampling
    if train_val_split:
        num, idx = train_val_split
        part_cnt = len(pairs) // num
        if phase == 'train':
            pairs = pairs[:part_cnt*idx] + pairs[part_cnt*(idx+1):]
        elif phase == 'val':
            pairs = pairs[part_cnt*idx : part_cnt*(idx+1)]
    print(phase, len(pairs))
    dataset = PPGDataset(pairs, phase, classes)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(phase=='train'),   
    )
    return dataloader

