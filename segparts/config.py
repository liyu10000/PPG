import argparse
import os

def str2bool(v):
    return v.lower() in ('true')

class Config:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):
        ### Common configuration.
        self._parser.add_argument('--seed', type=int, default=42, help='set the random seed for random, np, and torch')
        self._parser.add_argument('--image_dir', type=str, default='../data/labeled/images')
        self._parser.add_argument('--label_dir', type=str, default='../data/labeled/labels')
        self._parser.add_argument('--classes', type=int, default=6, choices=[6, 3, 1], help='number of classes')
        self._parser.add_argument('--model_path', type=str, default='./xcep_0323.pth')
        self._parser.add_argument('--num_workers', type=int, default=4)
        
        ### Training configuration.
        self._parser.add_argument('--train_batch_size', type=int, default=4, help='batch size for training')
        self._parser.add_argument('--val_batch_size', type=int, default=4, help='batch size for validation')
        self._parser.add_argument('--accumulation_steps', type=int, default=32, help='number of steps to update params')
        self._parser.add_argument('--lr', type=float, default=0.001)
        self._parser.add_argument('--num_epochs', type=int, default=90)
        self._parser.add_argument('--resume', type=str2bool, default=False)
        self._parser.add_argument('--resume_from', type=int, default=30, help='number of epochs to start counting')
        
        ### Testing configuration.
        self._parser.add_argument('--test_batch_size', type=int, default=4, help='batch size for testing')
        ## test on train
        # self._parser.add_argument('--test_image_dir', type=str, default='../data/labeled/images')
        # self._parser.add_argument('--test_label_dir', type=str, default='../data/labeled/labels')
        # self._parser.add_argument('--pred_mask_dir', type=str, default='../data/labeled/pred_masks')
        ## test on Hi-Resx2
        # self._parser.add_argument('--test_image_dir', type=str, default='../data/Hi-Resx2')
        # self._parser.add_argument('--test_label_dir', type=str, default='None')
        # self._parser.add_argument('--pred_mask_dir', type=str, default='../data/labeled/pred_masks/Hi-Resx2')
        ## test on separate testset (images/labels)
        self._parser.add_argument('--test_image_dir', type=str, default='../data/Segmentation_Test_Set/images')
        self._parser.add_argument('--test_label_dir', type=str, default='../data/Segmentation_Test_Set/labels')
        self._parser.add_argument('--pred_mask_dir', type=str, default='../data/Segmentation_Test_Set/pred_masks/')
        ## test on separate testset (images only)
        # self._parser.add_argument('--test_image_dir', type=str, default='../data/Segmentation_Test_Set/imagestest')
        # self._parser.add_argument('--test_label_dir', type=str, default='None')
        # self._parser.add_argument('--pred_mask_dir', type=str, default='../data/Segmentation_Test_Set/pred_maskstest')
        
        ### postprocessing and evaluation configuration.
        self._parser.add_argument('--plot_mask_dir', type=str, default='../data/Segmentation_Test_Set/plot_masks')
        # self._parser.add_argument('--plot_mask_dir', type=str, default='../data/Segmentation_Test_Set/plot_maskstest')

    def parse(self):
        cfg = self._parser.parse_args()
        return cfg
