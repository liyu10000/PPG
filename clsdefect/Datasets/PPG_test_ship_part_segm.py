import os
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import warnings
import tqdm
from typing import Union
from clsdefect.preprocessing.preprocessing_training import generate_3layer_mask
from clsdefect.utils import list_all_files_sorted, list_all_folders, tensor_show_write


classes = ['corrosion', 'fouling', 'delamination']
images_phrases = ["*HR.png", "*HR.jpg", "*SR.png", "*SR.jpg"]  # a keyword to get the image file


class PPGTestShipPartSegmList(Dataset):
    def __init__(self, roots: list, names: Union[list, str], ship_results_path: str, part_results_path: str, seg_results_path: str, normalize: bool):
        super().__init__()
        self.roots_list = roots
        self.classes = classes

        self.normalize = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]]) if normalize else None
        self.to_tensor = T.ToTensor()
        if names != 'new':
            self.list_folders = []
            list_names = np.array([False for _ in names], dtype=bool)
            for dir in self.roots_list:
                folders = list_all_files_sorted(dir)
                for f in folders:
                    for i, n in enumerate(names):
                        base = os.path.basename(f)
                        base = base.lower().replace('-', '_').replace('.0', '')
                        n = n.lower().replace('-', '_').replace('.0', '')
                        if base.startswith(n):
                            self.list_folders.append(f)
                            list_names[i] = True

            # list_names[[i for i, e in enumerate(names) if os.path.basename(e).startswith('ship_23_2019_image_34_x2_SR')]] = True
            assert list_names.all()
        else:
            self.list_folders = []
            list_names = np.array([False for _ in names], dtype=bool)
            for dir in self.roots_list:
                folders = list_all_files_sorted(dir)
                for f in folders:
                    self.list_folders.append(f)
        self.names = names
        self.length = len(self.list_folders)
        self.seg_results_files = list_all_files_sorted(seg_results_path)
        self.ship_results_files = list_all_files_sorted(ship_results_path)
        self.part_results_files = list_all_files_sorted(part_results_path)

        # sanity check to make sure that all defect segmentation, whole ship, and part segmentation files exist
        for folder in self.list_folders:
            seg_mask_file = self.get_mask(self.seg_results_files, folder)
            ship_mask_file = self.get_mask(self.ship_results_files, folder)
            part_mask_file = self.get_mask(self.part_results_files, folder)

        # self.list_folders = self.list_folders[38:39]
        # self.length = len(self.list_folders)

    def get_mask(self, results_files, folder):
        folder = os.path.splitext(os.path.basename(folder).lower().replace('-', '_').replace('.0', ''))[0]
        if self.names == 'new':
            x = [f for f in results_files if os.path.splitext(os.path.basename(f).lower().replace('-', '_').
                                                              replace('.0', ''))[0] == folder]
        else:
            x = [f for f in results_files if os.path.splitext(os.path.basename(f).lower().replace('-', '_').
                                                              replace('.0', ''))[0].startswith(folder)]
        if not len(x) == 1:
            print(x)
            print(folder)
        assert len(x) == 1
        return x[0]

    def get_input(self, folder):
        if self.names != 'new':
            flag = False
            image_file = None
            for idx, phrase in enumerate(images_phrases):
                if idx > 0:
                    warnings.warn("checking other phrases {}".format(phrase))
                if len(list_all_files_sorted(folder, phrase=phrase)) == 1:
                    flag = True
                    image_file = list_all_files_sorted(folder, phrase=phrase)[0]
                    break
            if not flag or not image_file:
                raise FileExistsError(f"that image doesn't exist (skipping): {folder}")
        else:
            image_file = folder

        # image = Image.open(image_file).convert("RGB")
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        return image

    def __getitem__(self, idx):
        folder = self.list_folders[idx]

        # get segmentation mask
        seg_mask_file = self.get_mask(self.seg_results_files, folder)
        seg_mask = self.to_tensor(Image.open(seg_mask_file).convert("1"))

        # get ship mask
        ship_mask_file = self.get_mask(self.ship_results_files, folder)
        ship_mask = self.to_tensor(Image.open(ship_mask_file).convert("1"))

        # get part mask
        part_mask_file = self.get_mask(self.part_results_files, folder)
        part_mask = self.to_tensor(Image.open(part_mask_file).convert("RGB"))

        # get input image
        image = self.get_input(folder)

        # get ground truth classes mask
        cls_mask = self.to_tensor(generate_3layer_mask(folder, image))
        image = self.to_tensor(image)

        if self.normalize:
            image_normalized = self.normalize(image)
            return image_normalized, ship_mask, part_mask, seg_mask, cls_mask, image, folder
        else:
            return image, ship_mask, part_mask, seg_mask, cls_mask, image, folder

    def __len__(self):
        return self.length


# if __name__ == "__main__":
#     batch_size = 1
#     seg_out_path = "/home/krm/ext/PPG/Classification_datasets/PPG/defects-seg-pred/hrsr_nobg_w10_90p_joint/"
#     inp_out_paths = [
#         "/home/krm/ext/PPG/Classification_datasets/PPG/AllHR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/AllSR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch2SR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch3HR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch4SR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch4HR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch5HR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch6HR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch6SR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch7HR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch8SR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch9HR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/batch9SR",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/HR_Test",
#         "/home/krm/ext/PPG/Classification_datasets/PPG/SR_Test",
#     ]
#     SR_names = [
#         'ship_10_2014_image_9_x2_SR',
#         'ship_7_2011_image_8_x2_SR',
#         'ship_26_2019_image_8_x2_SR',
#         'ship_12_2012_image_2_x2_SR',
#         'ship_22_2014_image_3_x2_SR',
#         'ship_23_2019_image_34_x2_SR',
#         'ship_24_2019_image_5_x2_SR',
#         'ship_16_2018_image_4_x2_SR',
#         'ship_20_2019_image_5_x2_SR',
#         'ship_5_2019_image_6_x2_SR',
#     ]
#     HR_names = [
#         '3HR',
#         '6HR',
#         '7HR',
#         '10HR',
#         '58HR',
#         '59HR',
#         '64HR',
#         '74HR',
#         '60HR',
#         '5HR',
#     ]
#     data = PPGTestList(roots=inp_out_paths, names=SR_names+HR_names, seg_results_path=seg_out_path, normalize=True)
#     loader = torch.utils.data.DataLoader(dataset=data, num_workers=4, batch_size=batch_size, shuffle=False)
#     for i, batch in enumerate(tqdm.tqdm(loader)):
#         img_norm, seg, cls, img = batch
#         out = torch.nn.functional.interpolate(torch.cat((img, seg.repeat(1, 3, 1, 1), cls), 3), size=(512,512*3))
#         tensor_show_write(out, name='batch', wait_value=5000, isCV=False)
