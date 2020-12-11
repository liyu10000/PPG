import os
import sys
import cv2
import time
import random
import warnings
import numpy as np
import pandas as pd
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
        df = df[df.train == 0]
        names = df.name.to_list()
        print(len(names), names)
        self.classes = cfg.classes
        self.whole_mask_dir = cfg.whole_mask_dir
        self.num_workers = cfg.num_workers
        self.batch_size = cfg.test_batch_size
        self.image_dir = cfg.test_image_dir
        self.label_dir = cfg.test_label_dir
        self.pred_npy_dir = cfg.pred_npy_dir
        os.makedirs(self.pred_npy_dir, exist_ok=True)
        self.device = torch.device('cuda:0')
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.backends.cudnn.benchmark = True
        # load weights
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
                                    names=names,
                                    image_dir=self.image_dir,
                                    label_dir=self.label_dir,
                                    whole_mask_dir=self.whole_mask_dir,
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
            # np.save(os.path.join(self.pred_npy_dir, name+'.npy'), prob)
            pred_mask = prob > 0.5  # convert to binary mask
            pred_mask = pred_mask.astype(np.uint8)
            pred_mask = pred_mask * 255
            pred_mask = pred_mask.transpose((1, 2, 0))
            cv2.imwrite(os.path.join(self.pred_img_dir, name+'.png'), pred_mask)

    def start(self):
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                names, images, _ = batch
                probs = self.forward(images)
                self.save(names, probs)
                print("Finished batch {}:".format(i), names)


if __name__ == '__main__':
    # aux_params=dict(
    #     pooling='max',             # one of 'avg', 'max'
    #     dropout=0.5,               # dropout ratio, default is None
    #     activation=None,           # activation function, default is None, can choose 'sigmoid'
    #     classes=4,                 # define number of output labels
    # )
    model = smp.Unet(cfg.backbone, 
                     in_channels=4 if len(cfg.whole_mask_dir)>1 else 3, 
                     classes=cfg.classes, 
                     encoder_weights=None, 
                     activation=None)

    tester = Tester(model, cfg)
    tester.start()
