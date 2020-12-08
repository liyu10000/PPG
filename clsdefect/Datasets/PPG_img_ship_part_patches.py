import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
from clsdefect.preprocessing.preprocessing_training import calculate_number_patches


classes = ['corrosion', 'fouling', 'delamination']
images_phrases = ["*HR.png", "*HR.jpg", "*SR.png", "*SR.jpg"]  # a keyword to get the image file


class PPGImgShipPartPatches(Dataset):
    def __init__(self, inp: torch.Tensor, ship_mask: torch.Tensor, part_mask: torch.Tensor, seg_mask: torch.Tensor,
                 cls_mask: torch.Tensor, img: torch.Tensor, patch_size: int = 64, stride_size: int = 32,
                 ratio_threshold: float = 0.1):
        super().__init__()
        assert len(inp.shape) == 3 and inp.shape[0] == 3
        assert len(seg_mask.shape) == 3 and seg_mask.shape[0] == 1
        assert len(ship_mask.shape) == 3 and ship_mask.shape[0] == 1
        assert len(part_mask.shape) == 3 and part_mask.shape[0] == 3
        assert len(cls_mask.shape) == 3 and cls_mask.shape[0] == 3
        assert seg_mask.shape[1:] == inp.shape[1:] == cls_mask.shape[1:] == ship_mask.shape[1:]
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.img = img
        self.inp = inp
        self.seg_mask = seg_mask
        self.ship_mask = ship_mask
        self.part_mask = part_mask
        self.cls_mask = cls_mask
        self.classes = classes
        self.ratio_th = ratio_threshold
        self.to_tensor = T.ToTensor()
        self.H, self.W = self.img.shape[-2:]
        self.nW, self.nH = calculate_number_patches(self.W, self.H, self.patch_size, self.stride_size)
        self.length = self.nW*self.nH

    def __getitem__(self, idx):
        iw = idx//self.nH
        ih = idx - iw*self.nH
        sl_h = slice(ih * self.stride_size, ih * self.stride_size + self.patch_size)
        sl_w = slice(iw * self.stride_size, iw * self.stride_size + self.patch_size)
        patch = self.inp[..., sl_h, sl_w]
        ship_patch = self.ship_mask[..., sl_h, sl_w]
        part_patch = self.part_mask[..., sl_h, sl_w]
        seg_patch = self.seg_mask[..., sl_h, sl_w]
        cls_patch = self.cls_mask[..., sl_h, sl_w]
        img_patch = self.img[..., sl_h, sl_w]
        label = cls_patch.mean(dim=(1, 2)) > self.ratio_th
        coord = torch.tensor([ih, iw])
        return patch, ship_patch, part_patch, seg_patch, cls_patch, img_patch, label, coord

    def __len__(self):
        return self.length
