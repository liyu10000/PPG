import os
import sys
import cv2
import time
import warnings
import numpy as np
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from .data import generator


class Tester(object):
    '''This class takes care of testing of our model'''
    def __init__(self, cfg):
        self.classes = cfg.classes
        self.num_workers = cfg.num_workers
        self.batch_size = cfg.test_batch_size
        self.image_dir = cfg.test_image_dir
        self.label_dir = cfg.test_label_dir
        self.pred_dir = cfg.pred_dir
        self.orisize_mask_dir = cfg.orisize_mask_dir
        self.orisize_pred_dir = cfg.orisize_pred_dir
        os.makedirs(self.pred_dir, exist_ok=True)
        if self.orisize_mask_dir:
            os.makedirs(self.orisize_pred_dir, exist_ok=True)
        self.device = torch.device('cuda:0')
        
        # load weights
        # aux_params=dict(
        #     pooling='max',             # one of 'avg', 'max'
        #     dropout=0.5,               # dropout ratio, default is None
        #     activation=None,           # activation function, default is None, can choose 'sigmoid'
        #     classes=4,                 # define number of output labels
        # )
        model = smp.Unet(cfg.backbone, 
                         in_channels=3, 
                         classes=cfg.classes, 
                         encoder_weights=None, 
                         activation=None)
        self.net = model
        best_model = cfg.model_path
        if not os.path.isfile(best_model):
            print('*****WARNING*****: {} does not exist.'.format(best_model))
            sys.exit()
        checkpoint = torch.load(best_model)
        # self.epoch = checkpoint["epoch"]
        # self.best_loss = checkpoint["loss"]
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net = self.net.to(self.device)
        self.net.eval()
        # initiate data loader
        self.dataloader = generator(
                                    image_dir=self.image_dir,
                                    label_dir=self.label_dir,
                                    phase="test",
                                    classes=self.classes,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    )
        
    def forward(self, images):
        images = images.to(self.device)
        outputs = self.net(images)
        if self.classes > 1:
            probs = F.softmax(outputs, dim=1)
        else:
            probs = torch.sigmoid(outputs)
        probs = probs.cpu().numpy()
        return probs

    def save(self, names, probs):
        for name, prob in zip(names, probs):
            pred_mask = prob > 0.5  # convert to binary mask
            pred_mask = pred_mask.astype(np.uint8)
            pred_mask = pred_mask * 255
            pred_mask = pred_mask.transpose((1, 2, 0))
            cv2.imwrite(os.path.join(self.pred_dir, name+'.png'), pred_mask)
            if self.orisize_mask_dir:
                ori_mask = cv2.imread(os.path.join(self.orisize_mask_dir, name+'.png'))
                H, W, _ = ori_mask.shape
                pred_mask = cv2.resize(pred_mask, (W, H))
                cv2.imwrite(os.path.join(self.orisize_pred_dir, name+'.png'), pred_mask)

    def start(self):
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                names, images, _ = batch
                probs = self.forward(images)
                self.save(names, probs)
                print("Finished batch {}:".format(i), names)


if __name__ == '__main__':

    tester = Tester(cfg)
    tester.start()
