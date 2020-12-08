from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import clsdefect.transforms_image_mask_label as T
from clsdefect.preprocessing.preprocessing_training import get_label
import tqdm
from clsdefect.utils import list_all_files_sorted, list_all_folders, load_obj

classes = ['corrosion', 'fouling', 'delamination']


class PPGTrain2(Dataset):
    def __init__(self, root: str, normalize: bool, area_threshold: float, ratio_threshold: float, percentages: str or None = None, patch_size: tuple = (64, 30)):
        if percentages:
            percentages = [int(i) for i in percentages.split('_')]
            assert len(percentages) == 2
            assert sum(percentages) == 100
        assert 0 <= ratio_threshold <= 1
        super().__init__()
        self.root = root
        self.classes = classes
        self.ratio_th = ratio_threshold
        self.area_threshold = area_threshold

        transforms = []
        scale = np.floor(patch_size[0] // patch_size[1])
        pad = int((patch_size[0] - scale * patch_size[1]) // 2)
        if scale != 1:
            transforms.append(T.Resize(int(scale * patch_size[1])))
        if pad != 0:
            transforms.append(T.Pad(padding=pad))

        transforms.append(T.ToTensorEncodeLabels(len(self.classes)))
        if normalize:
            transforms.append(T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]]))

        self.transforms = T.Compose(transforms)
        self.list_patches = list_all_files_sorted(self.root, "*_patch.png")
        self.list_masks = list_all_files_sorted(self.root, "*_mask.png")
        self.list_metas = list_all_files_sorted(self.root, "*_meta.pkl")
        # print(self.list_masks[0])
        # assert len(self.list_masks) == len(self.list_patches) == len(self.list_metas)
        self.length = len(self.list_masks)
        if percentages:
            self.l_train = round(self.length*percentages[0]/100)
            self.l_val = self.length - self.l_train
            if percentages[1] > 0:
                assert self.l_train != 0 and self.l_val != 0

    def __getitem__(self, idx):
        mask_name = self.list_masks[idx]
        meta_name = mask_name.replace('_mask.png', '_meta.pkl')
        image_name = mask_name.replace('_mask.png', '_patch.png')
        image = Image.open(image_name).convert("RGB")
        meta = load_obj(meta_name)
        # masks = [m.convert("1") for m in Image.open(mask_name).split()]
        labels = list(set([get_label(overlap['label']) for overlap in meta['overlapping']
                           if overlap['ratio'] >= self.ratio_th or overlap['area'] >= self.area_threshold]))
        # masks = [overlap['mask'] for overlap in meta['overlapping'] if overlap['label'] == 2 and (overlap['ratio'] >= self.ratio_th or overlap['area'] >= self.area_threshold)]
        # masks = combine_images(masks)
        # masks = None
        masks = Image.open(mask_name)
        if self.transforms is not None:
            image, labels, masks = self.transforms(image, labels, masks)

        return image, labels, masks

    def __len__(self):
        return self.length


if __name__ == "__main__":
    batch_size = 20
    data = PPGTrain2("/home/krm/ext/PPG/Classification_datasets/PPG/NewTest_32/",
                     normalize=True, area_threshold=0, ratio_threshold=0.1)
    loader = torch.utils.data.DataLoader(dataset=data, num_workers=4, batch_size=batch_size, shuffle=False)
    names = torch.zeros([len(data), 3])
    for i, batch in enumerate(tqdm.tqdm(loader)):
        patches, labels = batch
        names[i*batch_size: (i+1)*batch_size] = labels
        # continue
    sum = names.sum(dim=0)
    average = names.mean(dim=0)
    print(sum)
    print(average)
    print('done')
