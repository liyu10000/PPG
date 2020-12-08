import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import clsdefect.transforms_image_mask_label as T
from clsdefect.preprocessing.preprocessing_training import get_label
import tqdm
from clsdefect.utils import list_all_folders, list_all_files_sorted, load_obj


classes = ['corrosion', 'fouling', 'delamination']


class PPGTrainList(Dataset):
    def __init__(self, root_list: list, normalize: bool, area_threshold: float, ratio_threshold: float,
                 percentages: str or None = None, patch_size: list = [(64, 64), (64, 32)],
                 except_list: list or None = None):
        if percentages:
            percentages = [int(i) for i in percentages.split('_')]
            assert len(percentages) == 2
            assert sum(percentages) == 100

        assert 0 <= ratio_threshold <= 1
        super().__init__()
        self.root_list = root_list
        self.classes = classes
        self.ratio_th = ratio_threshold
        self.area_threshold = area_threshold

        transforms = []
        for pair in patch_size:
            transforms.append([])
            scale = np.floor(pair[0] // pair[1])
            pad = int((pair[0] - scale * pair[1]) // 2)
            if scale != 1:
                transforms[-1].append(T.Resize(int(scale * pair[1])))
            if pad != 0:
                transforms[-1].append(T.Pad(padding=pad))
            transforms[-1].append(T.RandomHorizontalFlip(0.5))
            transforms[-1].append(T.RandomVerticalFlip(0.5))
            transforms[-1].append(T.ToTensorEncodeLabels(len(self.classes)))
            if normalize:
                transforms[-1].append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transforms = [T.Compose(trans) for trans in transforms]
        self.dirs_per_trans = len(self.root_list) // len(self.transforms)
        assert len(self.root_list) == self.dirs_per_trans * len(self.transforms)
        self.list_masks = []
        self.length_stairs = [0]
        length = 0

        if except_list:
            list_found = np.array([False for _ in except_list], dtype=bool)

        for dir in self.root_list:
            files_origianl = list_all_files_sorted(dir, "*_mask.png")
            if except_list:
                files = []
                for f in files_origianl:
                    # print(os.path.basename(f))
                    for i, e in enumerate(except_list):
                        if e.startswith('ship'):
                            e = e.split('_')
                            e = '_'.join(e[1:-2])
                        if e in os.path.basename(f):
                            list_found[i] = True
                    if not list_found[i]:
                        files.append(f)
            else:
                files = files_origianl

            self.list_masks.append(files)
            length += len(self.list_masks[-1])
            self.length_stairs.append(length)

        if except_list:
            list_found[[i for i, e in enumerate(except_list) if
                        os.path.basename(e).startswith('ship_23_2019_image_34_x2_SR') or
                        os.path.basename(e).startswith('ship_9_2013_image_20_x2_SR') or
                        os.path.basename(e).startswith('ship_5_2019_image_22_x2_SR') or
                        os.path.basename(e).startswith('ship_20_2011_image_24_x2_SR')]] = True
            [print(except_list[i]) for i in range(len(except_list)) if not list_found[i]]
            assert list_found.all()

        if percentages:
            self.l_train = round(length*percentages[0]/100)
            self.l_val = length - self.l_train
            if percentages[1] > 0:
                assert self.l_train != 0 and self.l_val != 0
            assert self.l_val + self.l_train == self.length_stairs[-1]

    def __getitem__(self, idx):
        for big_idx, length in enumerate(self.length_stairs):
            if idx < length:
                break

        big_idx = big_idx - 1
        small_idx = idx - self.length_stairs[big_idx]
        trans_idx = big_idx // self.dirs_per_trans

        mask_name = self.list_masks[big_idx][small_idx]
        meta_name = mask_name.replace('_mask.png', '_meta.pkl')
        image_name = mask_name.replace('_mask.png', '_patch.png')
        image = Image.open(image_name).convert("RGB")
        meta = load_obj(meta_name)
        labels = list(set([get_label(overlap['label']) for overlap in meta['overlapping']
                           if overlap['ratio'] >= self.ratio_th or overlap['area'] >= self.area_threshold]))

        masks = Image.open(mask_name)
        if self.transforms is not None:
            image, labels, masks = self.transforms[trans_idx](image, labels, masks)

        return image, labels, masks

    def __len__(self):
        return self.length_stairs[-1]


if __name__ == "__main__":
    batch_size = 20
    data = PPGTrainList(["/home/krm/ext/PPG/Classification_datasets/PPG/NewTest_32/"],
                        normalize=True, area_threshold=0, ratio_threshold=0.1)
    loader = torch.utils.data.DataLoader(dataset=data, num_workers=4, batch_size=batch_size, shuffle=False)
    names = torch.zeros([len(data), 3])
    for i, batch in enumerate(tqdm.tqdm(loader)):
        patches, labels = batch
        names[i*batch_size: (i+1)*batch_size] = labels

    sum = names.sum(dim=0)
    average = names.mean(dim=0)
    print(sum)
    print(average)
    print('done')
