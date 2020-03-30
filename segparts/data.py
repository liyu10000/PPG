import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
from albumentations import RandomContrast, RandomBrightness, RandomBrightnessContrast, ToGray
from albumentations.pytorch import ToTensor
from config import Config

cfg = Config().parse()
seed = cfg.seed
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)


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


def resize_without_pad(img, W, H):
    h, w, _ = img.shape
    h_factor = H / h
    w_factor = W / w
    img = cv2.resize(img, (W, H))
    return img, w_factor, h_factor

def make_mask(label_info, class_index, w_factor, h_factor, W, H):
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
            points[:, 0] *= w_factor
            points[:, 1] *= h_factor
            cv2.fillConvexPoly(mask[i, :, :], points.astype(int), 1.0)  # mask[:, :, i] doesn't work
    mask = mask.transpose(1, 2, 0) # H, W, C
    return mask


def get_transforms(phase, mean, std):
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


### Dataloader
class PPGDataset(Dataset):
    def __init__(self, label_dict, class_index, mean, std, phase, W, H, pad):
        self.names = list(label_dict.keys())
        self.label_dict = label_dict
        self.class_index = class_index
        self.classes = len(set(class_index.values()))
        self.mean = mean
        self.std = std
        self.phase = phase
        self.W = W
        self.H = H
        self.pad = pad # if pad at resize
        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        name = self.names[idx]
        label_info = self.label_dict[name]
        img = cv2.imread(label_info["path"])
        if self.pad:
            img, factor, direction, padding = resize_with_pad(img, self.W, self.H)
        else:
            img, w_factor, h_factor = resize_without_pad(img, self.W, self.H)
        if self.phase == "test":
            H, W, _ = img.shape
            C = self.classes
            mask = np.zeros((H, W, C))
        else:
            if self.pad:
                mask = make_mask(label_info, self.class_index, factor, direction, padding, self.W, self.H)
            else:
                mask = make_mask(label_info, self.class_index, w_factor, h_factor, self.W, self.H)
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
            classes,
            W,
            H,
            pad,
            val_interval=[0,12],
            mean=None,
            std=None,
            shuffle=True,
            batch_size=8,
            num_workers=4,
            ):
    label_dict = get_label_dict(image_dir, label_dir)
    if classes == 6:
        class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':3, 'PS BT':4, 'PS VS':5}
    elif classes == 3:
        class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':0, 'PS BT':1, 'PS VS':2}
    else:  # classes = 1
        class_index = {'STBD TS':0, 'STBD BT':0, 'STBD VS': 0, 'PS TS':0, 'PS BT':0, 'PS VS':0}
    keys = list(label_dict.keys())
    keys.sort()
    random.Random(seed).shuffle(keys) # shuffle with seed, so that yielding same sampling
    val_interval1, val_interval2 = val_interval
    if phase == "train":
        sample_keys = keys[:val_interval1] + keys[val_interval2:]
    elif phase == "val":
        sample_keys = keys[val_interval1:val_interval2]
    else:
        sample_keys = keys[:12]
    # sample_keys = keys  # use all data for train & val
    print(phase, len(sample_keys), sample_keys)
    sample_label_dict = {key:label_dict[key] for key in sample_keys}
    dataset = PPGDataset(sample_label_dict, class_index, mean, std, phase, W, H, pad)
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
    image_dir = "../data/labeled/images"
    label_dir = "../data/labeled/labels"
    label_dict = get_label_dict(image_dir, label_dir)
    class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':0, 'PS BT':1, 'PS VS':2}
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
    phase = "train"
    dataset = PPGDataset(label_dict, class_index, mean, std, phase, 640, 480, False)

    tmp_dir = './tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    print('# files', len(dataset))
    for i in range(len(dataset)):
        name, img, mask = dataset[i]
        print(name, img.shape, mask.shape)
        
        img = img.numpy().transpose((1, 2, 0))
        mask = mask.numpy().transpose((1, 2, 0))
        img *= 255
        mask *= 255
        cv2.imwrite(os.path.join(tmp_dir, name+'.jpg'), img)
        cv2.imwrite(os.path.join(tmp_dir, name+'_mask.jpg'), mask)
        # break

    # # calculate mean and std of dataset
    # mean, std = calc_mean_std(dataset)
    # print('mean: ', mean)
    # print('std: ', std)