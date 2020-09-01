import os
import sys
import cv2
import time
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from config import Config
from data import generator


cfg = Config().parse()
print(cfg)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)


class Tester(object):
    '''This class takes care of testing of our model'''
    def __init__(self, model, cfg):
        df = pd.read_csv(cfg.names_file)
        df = df[df[cfg.testkey] == 1]
        names = df.name.to_list()
        print(len(names), names)
        self.num_workers = cfg.num_workers
        self.batch_size = cfg.test_batch_size
        self.image_dir = cfg.test_image_dir
        self.label_dir = cfg.test_label_dir if cfg.test_label_dir != 'None' else None
        self.pred_mask_dir = cfg.pred_mask_dir
        os.makedirs(self.pred_mask_dir, exist_ok=True)
        self.device = torch.device('cuda:0')
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.backends.cudnn.benchmark = True
        self.classes = cfg.classes
        # load weights
        self.net = model
        best_model = cfg.model_path
        if not os.path.isfile(best_model):
            print('*****WARNING*****: {} does not exist.'.format(best_model))
            sys.exit()
        checkpoint = torch.load(best_model, map_location='cuda:0')
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["loss"]
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net = self.net.to(self.device)
        self.net.eval()
        # initiate data loader
        self.dataloader = generator(
                                    names=names,
                                    image_dir=self.image_dir,
                                    label_dir=self.label_dir,
                                    phase="test",
                                    classes=self.classes,
                                    weight=1.0,
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
            np.save(os.path.join(self.pred_mask_dir, name+'.npy'), prob)
            # pred_mask = prob > 0.5  # convert to binary mask
            # pred_mask = pred_mask.astype(np.uint8)
            # pred_mask = pred_mask * 255
            # pred_mask = pred_mask.transpose((1, 2, 0))
            # cv2.imwrite(os.path.join(self.pred_mask_dir, name+'.png'), pred_mask)

    def start(self):
        with torch.no_grad():
            for batch in tqdm(self.dataloader, ncols=100):
                names, images, _, _ = batch
                probs = self.forward(images)
                self.save(names, probs)


if __name__ == '__main__':
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

    tester = Tester(model, cfg)
    tester.start()
