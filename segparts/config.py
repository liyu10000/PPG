import argparse
import os


class Config:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):
        ### Common configuration.
        self._parser.add_argument('--seed', type=int, default=42, help='set the random seed for random, np, and torch')
        self._parser.add_argument('--image_dir', type=str, default='../dataparts/labeled/images_3cls')
        self._parser.add_argument('--label_dir', type=str, default='../dataparts/labeled/labels_3cls')
        self._parser.add_argument('--classes', type=int, default=3, choices=[6, 3, 1], help='number of classes')
        self._parser.add_argument('--model_path', type=str, default='./model.pth')
        self._parser.add_argument('--num_workers', type=int, default=4)
        
        ### Training configuration.
        self._parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'bce_dice', 'dice'])
        self._parser.add_argument('--train_batch_size', type=int, default=2, help='batch size for training')
        self._parser.add_argument('--val_batch_size', type=int, default=2, help='batch size for validation')
        self._parser.add_argument('--accumulation_steps', type=int, default=16, help='number of steps to update params')
        self._parser.add_argument('--lr', type=float, default=0.0001)
        self._parser.add_argument('--num_epochs', type=int, default=30)
        self._parser.add_argument('--save', type=str, default='best', choices=['best', 'all'])
        self._parser.add_argument('--resume', type=str, default='False', choices=['True', 'False'])
        self._parser.add_argument('--resume_from', type=int, default=0, help='number of epochs to start counting')
        
        ### Testing configuration.
        self._parser.add_argument('--test_batch_size', type=int, default=4, help='batch size for testing')
        
        ## test on train
        self._parser.add_argument('--test_image_dir', type=str, default='../dataparts/labeled/images')
        self._parser.add_argument('--test_label_dir', type=str, default='../dataparts/labeled/labels')
        self._parser.add_argument('--pred_mask_dir', type=str, default='../dataparts/labeled/pred_masks')
        self._parser.add_argument('--plot_mask_dir', type=str, default='../dataparts/labeled/plot_masks')
        
        ## test on separate testset (images/labels)
        # self._parser.add_argument('--test_image_dir', type=str, default='../dataparts/Segmentation_Test_Set/images')
        # self._parser.add_argument('--test_label_dir', type=str, default='../dataparts/Segmentation_Test_Set/labels')
        # self._parser.add_argument('--pred_mask_dir', type=str, default='../dataparts/Segmentation_Test_Set/pred_masks')
        # self._parser.add_argument('--plot_mask_dir', type=str, default='../dataparts/Segmentation_Test_Set/plot_masks')
    

    def parse(self):
        cfg = self._parser.parse_args()
        return cfg
