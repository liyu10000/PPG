import os
import cv2
import time
import random
import warnings
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import generator
from metrics import Meter


warnings.filterwarnings("ignore")
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


image_dir = "../data/labeled/images"
label_dir = "../data/labeled/labels"
best_model = "./xcep_model.pth"


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model):
        self.num_workers = 8
        self.batch_size = {"train": 8, "val": 8}
        self.accumulation_steps = 64 // self.batch_size['train']
        self.lr = 1e-3
        self.resume = True
        self.num_epochs = 60
        self.epoch = 0
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        if self.resume:
            checkpoint = torch.load(best_model)
            self.epoch = checkpoint["epoch"] + 1  # it may not be the last epoch being runned
            # self.epoch = 40 # the last epoch number
            self.best_loss = checkpoint["best_loss"]
            self.net.load_state_dict(checkpoint["state_dict"])
        self.net = self.net.to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.dataloaders = {
            phase: generator(
                image_dir=image_dir,
                label_dir=label_dir,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        
    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.net(images)
        # print(images.size(), targets.size(), outputs.size())
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter()
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | Time: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
#         tk0 = tqdm_notebook(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
            _, images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
#             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        meter.log(epoch_loss)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        while self.epoch < self.num_epochs:
            self.iterate(self.epoch, "train")
            state = {
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                # "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(self.epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, best_model)
            self.epoch += 1
            print()



if __name__ == "__main__":
    # aux_params=dict(
    #     pooling='max',             # one of 'avg', 'max'
    #     dropout=0.5,               # dropout ratio, default is None
    #     activation=None,           # activation function, default is None, can choose 'sigmoid'
    #     classes=4,                 # define number of output labels
    # )
    model = smp.Unet("xception", in_channels=3, classes=6, encoder_weights="imagenet", activation=None)

    trainer = Trainer(model)
    trainer.start()

