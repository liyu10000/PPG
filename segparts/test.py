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

from data import generator


classes = 6
image_dir = "../data/labeled/images"
label_dir = "../data/labeled/labels"
pred_mask_dir = "../data/labeled/pred_masks"
best_model = "./xcep_tv_90th.pth"



def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


class Tester(object):
    '''This class takes care of testing of our model'''
    def __init__(self, model):
        self.num_workers = 8
        self.batch_size = 8
        self.pred_mask_dir = pred_mask_dir
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.backends.cudnn.benchmark = True
        self.classes = classes
        # load weights
        self.net = model
        if not os.path.isfile(best_model):
            print('*****WARNING*****: {} does not exist.'.format(best_model))
            sys.exit()
        checkpoint = torch.load(best_model)
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net = self.net.to(self.device)
        self.net.eval()
        # initiate data loader
        self.dataloader = generator(
                                    image_dir=image_dir,
                                    label_dir=label_dir,
                                    phase="test",
                                    # mean=(0.485, 0.456, 0.406), # statistics from ImageNet
                                    # std=(0.229, 0.224, 0.225),
                                    mean=(0.400, 0.413, 0.481),   # statistics from custom dataset
                                    std=(0.286, 0.267, 0.286),
                                    shuffle=False,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    )
        
    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
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

    def start(self):
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                names, images, targets = batch
                probs = self.forward(images, targets)
                self.save(names, probs)
                print("Finished batch {}:".format(i), names)
                break


if __name__ == '__main__':
    # aux_params=dict(
    #     pooling='max',             # one of 'avg', 'max'
    #     dropout=0.5,               # dropout ratio, default is None
    #     activation=None,           # activation function, default is None, can choose 'sigmoid'
    #     classes=4,                 # define number of output labels
    # )
    model = smp.Unet("xception", 
                     in_channels=3, 
                     classes=classes, 
                     encoder_weights=None, 
                     activation=None)

    tester = Tester(model)
    tester.start()
