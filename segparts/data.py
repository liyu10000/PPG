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


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5),
                # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                # RandomContrast(p=0.5),
                # RandomBrightness(p=0.5),
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
    def __init__(self, names, image_dir, label_dir, phase, mean, std):
        self.names = names
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(self.image_dir, name+'.png'))
        mask = cv2.imread(os.path.join(self.label_dir, name+'.png'))
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1xHxWxC
        mask = mask[0].permute(2, 0, 1) # CxHxW
        return name, img, mask

    def __len__(self):
        return len(self.names)


def generator(
            image_dir,
            label_dir,
            phase,
            val_interval,
            mean,
            std,
            shuffle=True,
            batch_size=8,
            num_workers=4,
            ):
    keys = [f[:-4] for f in os.listdir(image_dir) if f.endswith('.png')]
    keys.sort()
    random.Random(seed).shuffle(keys) # shuffle with seed, so that yielding same sampling
    val_interval1, val_interval2 = val_interval
    if phase == "train":
        sample_keys = keys[:val_interval1] + keys[val_interval2:]
    elif phase == "val":
        sample_keys = keys[val_interval1:val_interval2]
    else:
        sample_keys = keys[val_interval1:val_interval2]
    # sample_keys = keys  # use all data for train & val
    print(phase, len(sample_keys), sample_keys)
    dataset = PPGDataset(sample_keys, image_dir, label_dir, phase, mean, std)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,   
    )
    return dataloader



if __name__ == "__main__":
    # test dataloader
    image_dir = "../data/labeled/images_3cls"
    label_dir = "../data/labeled/labels_3cls"
    phase = "val"
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
    keys = [f[:-4] for f in os.listdir(image_dir) if f.endswith('.png')]
    dataset = PPGDataset(keys, image_dir, label_dir, phase, mean, std)

    # # output mask as img for checking
    # tmp_dir = './tmp'
    # os.makedirs(tmp_dir, exist_ok=True)
    # print('# files', len(dataset))
    # for i in range(len(dataset)):
    #     name, img, mask = dataset[i]
    #     print(name, img.shape, mask.shape)
        
    #     img = img.numpy().transpose((1, 2, 0))
    #     mask = mask.numpy().transpose((1, 2, 0))
    #     img *= 255
    #     mask *= 255
    #     cv2.imwrite(os.path.join(tmp_dir, name+'.jpg'), img)
    #     cv2.imwrite(os.path.join(tmp_dir, name+'_mask.jpg'), mask)
    #     break

    # # calculate mean and std of dataset
    # mean, std = calc_mean_std(dataset)
    # print('mean: ', mean)
    # print('std: ', std)