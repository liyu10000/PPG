import os
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

from .data import generator
from .loss import bce_loss, dice_loss, bce_dice_loss
from .log import Meter


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, cfg):
        df = pd.read_csv(cfg.names_file)
        df = df[df[cfg.trainkey[0]] == cfg.trainkey[1]]
        if cfg.onlySR == 'yes':
            df = df[df.resolution == 'SR']
        names = df.name.to_list()
        if cfg.takefirst != -1:
            # random.Random(cfg.seed).shuffle(names)
            names = names[:cfg.takefirst]
        # print(len(names), names)
        self.classes = cfg.classes
        self.image_dir = cfg.train_image_dirs
        self.label_dir = cfg.train_label_dirs
        self.num_workers = cfg.num_workers
        self.batch_size = {"train": cfg.train_batch_size, "val": cfg.val_batch_size}
        self.accumulation_steps =  cfg.accumulation_steps // self.batch_size['train']
        self.resume = cfg.resume
        self.num_epochs = cfg.num_epochs
        self.epoch = cfg.resume_from # the epoch to start counting
        os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
        self.save_all = True if cfg.save == 'all' else False # whether to save all epoch weights or just the best
        self.save_prefix = os.path.splitext(cfg.model_path)[0]
        self.best_model = cfg.model_path
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device('cuda:0')

        # aux_params=dict(
        #     pooling='max',             # one of 'avg', 'max'
        #     dropout=0.5,               # dropout ratio, default is None
        #     activation=None,           # activation function, default is None, can choose 'sigmoid'
        #     classes=4,                 # define number of output labels
        # )
        model = smp.Unet(cfg.backbone, 
                         in_channels=3, 
                         classes=cfg.classes, 
                         encoder_weights="imagenet", 
                         activation=None)

        self.net = model
        if self.resume:
            checkpoint = torch.load(cfg.model_path)
            # self.epoch = checkpoint["epoch"] + 1  # it may not be the last epoch being runned
            self.best_loss = checkpoint["loss"]
            self.net.load_state_dict(checkpoint["state_dict"])
            print('loaded {}, current loss: {}'.format(cfg.model_path, self.best_loss))
        self.net = self.net.to(self.device)
        self.weight = cfg.weight
        self.train_val_split = cfg.train_val_split
        if cfg.loss == 'bce':
            self.criterion = bce_loss
        elif cfg.loss == 'bce_dice':
            self.criterion = bce_dice_loss
        else:
            self.criterion = dice_loss
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=cfg.lr_patience, verbose=True)
        self.dataloaders = {
            phase: generator(
                names=names,
                image_dir=self.image_dir,
                label_dir=self.label_dir,
                phase=phase,
                classes=self.classes,
                weight=self.weight,
                train_val_split=self.train_val_split,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        
    def forward(self, images, targets, weights):
        """
        @params
            images: N,C,H,W
            targets: N,C,H,W
            weights: N,C,H,W
        """
        images = images.to(self.device)
        targets = targets.to(self.device)
        weights = weights.to(self.device)
        outputs = self.net(images)
        if self.weight != 1.0:
            loss = self.criterion(outputs, targets, reduction='none')
            loss = loss * weights
            loss = loss.mean()
        else:
            loss = self.criterion(outputs, targets)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter()
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | Time: {start}")
        self.net.train(phase == "train")
        batch_size = self.batch_size[phase]
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        # tk0 = tqdm_notebook(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
            _, images, targets, weights = batch
            loss, outputs = self.forward(images, targets, weights)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
            # tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        meter.log(epoch_loss)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        while self.epoch < self.num_epochs:
            self.iterate(self.epoch, "train")
            with torch.no_grad():
                val_loss = self.iterate(self.epoch, "val")
                self.scheduler.step(val_loss)
            state = {
                "epoch": self.epoch,
                "loss": val_loss,
                "state_dict": self.net.state_dict(),
                # "optimizer": self.optimizer.state_dict(),
            }
            if self.save_all:
                torch.save(state, '{}_{}.pth'.format(self.save_prefix, str(self.epoch).zfill(3)))
            if val_loss < self.best_loss:
                print("******** New optimal found, saving best model ********")
                self.best_loss = val_loss
                torch.save(state, self.best_model)
            self.epoch += 1
            print()



if __name__ == "__main__":
    from config import cfg

    trainer = Trainer(cfg)
    trainer.start()
