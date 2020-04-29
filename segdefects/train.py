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

from config import Config
from data import generator
from loss import bce_loss, dice_loss, bce_dice_loss
from log import Meter

warnings.filterwarnings("ignore")
cfg = Config().parse()
print(cfg)
seed = cfg.seed
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, cfg):
        self.image_dir = cfg.image_dir
        self.label_dir = cfg.label_dir
        self.num_workers = cfg.num_workers
        self.batch_size = {"train": cfg.train_batch_size, "val": cfg.val_batch_size}
        self.accumulation_steps =  cfg.accumulation_steps // self.batch_size['train']
        self.lr = cfg.lr
        self.resume = True if cfg.resume == 'True' else False
        self.num_epochs = cfg.num_epochs
        self.epoch = cfg.resume_from # the epoch to start counting
        os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
        self.save_all = True if cfg.save == 'all' else False # whether to save all epoch weights or just the best
        self.save_prefix = os.path.splitext(cfg.model_path)[0]
        self.best_model = cfg.model_path
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device('cuda:0')
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        if self.resume:
            checkpoint = torch.load(cfg.model_path)
            # self.epoch = checkpoint["epoch"] + 1  # it may not be the last epoch being runned
            self.best_loss = checkpoint["loss"]
            self.net.load_state_dict(checkpoint["state_dict"])
            print('loaded {}, current loss: {}'.format(cfg.model_path, self.best_loss))
        self.net = self.net.to(self.device)
        if cfg.loss == 'bce':
            self.criterion = bce_loss
        elif cfg.loss == 'bce_dice':
            self.criterion = bce_dice_loss
        else:
            self.criterion = dice_loss
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.dataloaders = {
            phase: generator(
                image_dir=self.image_dir,
                label_dir=self.label_dir,
                phase=phase,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        
    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.net(images)
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
        # tk0 = tqdm_notebook(dataloader, total=total_batches)
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
    # aux_params=dict(
    #     pooling='max',             # one of 'avg', 'max'
    #     dropout=0.5,               # dropout ratio, default is None
    #     activation=None,           # activation function, default is None, can choose 'sigmoid'
    #     classes=4,                 # define number of output labels
    # )
    model = smp.Unet("xception", 
                     in_channels=3, 
                     classes=cfg.classes, 
                     encoder_weights="imagenet", 
                     activation=None)

    trainer = Trainer(model, cfg)
    trainer.start()

