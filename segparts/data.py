import os
import cv2
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
from albumentations.pytorch import ToTensor


seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)


# ### RLE-Mask utility functions
# #https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
# def mask2rle(img):
#     '''
#     img: numpy array, 1 -> mask, 0 -> background
#     Returns run length as string formated
#     '''
#     pixels= img.T.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)

# def make_mask(row_id, df):
#     '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
#     fname = df.iloc[row_id].name
#     labels = df.iloc[row_id][:4]
#     masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
#     # 4:class 1～4 (ch:0～3)

#     for idx, label in enumerate(labels.values):
#         if label is not np.nan:
#             label = label.split(" ")
#             positions = map(int, label[0::2])
#             length = map(int, label[1::2])
#             mask = np.zeros(256 * 1600, dtype=np.uint8)
#             for pos, le in zip(positions, length):
#                 mask[pos:(pos + le)] = 1
#             masks[:, :, idx] = mask.reshape(256, 1600, order='F')
#     return fname, masks



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
    labels = scan_files(label_dir, postfix='.csv')
    label_dict = {os.path.splitext(f)[0]:{'path':os.path.join(image_dir, f)} for f in images}
    for f in labels:
        base_f = os.path.basename(f)
        tokens = base_f.split()
        name = tokens[0] + ' ' + tokens[1].split('_')[0]
        side = tokens[1].split('_')[1]
        part = tokens[2]
        key = side + ' ' + part
        if not name in label_dict:
            continue
        if not key in label_dict[name]:
            label_dict[name][key] = []
        label_dict[name][key].append(f)

    # pprint(label_dict)
    return label_dict

def resize_with_pad(img, W=640, H=480):
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

def make_mask(label_info, class_index, factor, direction, pad, W=640, H=480):
    """ 
    :param name: image name, like 'V9 50HR'
    :param class_index: {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':3, 'PS BT':4, 'PS VS':5}
    """
    class_num = len(class_index)
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

def get_transforms(phase, mean, std):
    list_transforms = []
    # if phase == "train":
    #     list_transforms.extend(
    #         [
    #             HorizontalFlip(p=0.5), # only horizontal flip as of now
    #         ]
    #     )
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
    def __init__(self, label_dict, class_index, mean, std, phase):
        self.names = list(label_dict.keys())
        self.label_dict = label_dict
        self.class_index = class_index
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        name = self.names[idx]
        label_info = self.label_dict[name]
        img = cv2.imread(label_info["path"])
        img, factor, direction, pad = resize_with_pad(img)
        mask = make_mask(label_info, self.class_index, factor, direction, pad)

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
            mean=None,
            std=None,
            shuffle=True,
            batch_size=8,
            num_workers=4,
            ):
    label_dict = get_label_dict(image_dir, label_dir)
    class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':3, 'PS BT':4, 'PS VS':5}
    keys = list(label_dict.keys())
    keys.sort()
    random.Random(seed).shuffle(keys) # shuffle with seed, so that yielding same sampling
    if phase == "train":
        sample_keys = keys[8:]
    elif phase == "val":
        sample_keys = keys[:8]
    else:
        sample_keys = keys
    sample_keys = keys  # use all data for train & val
    sample_label_dict = {key:label_dict[key] for key in sample_keys}
    dataset = PPGDataset(sample_label_dict, class_index, mean, std, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,   
    )
    return dataloader



if __name__ == "__main__":
    image_dir = "../data/labeled/images"
    label_dir = "../data/labeled/labels"
    label_dict = get_label_dict(image_dir, label_dir)
    class_index = {'STBD TS':0, 'STBD BT':1, 'STBD VS': 2, 'PS TS':3, 'PS BT':4, 'PS VS':5}
    mean = None
    std = None
    phase = "train"
    dataset = PPGDataset(label_dict, class_index, mean, std, phase)

    img, mask = dataset[0]
    print(img.shape, mask.shape)
