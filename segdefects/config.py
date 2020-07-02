import argparse
import os


class Config:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):
        ### Common configuration.
        self._parser.add_argument('--seed', type=int, default=42, help='set the random seed for random, np, and torch')
        self._parser.add_argument('--gpu', type=int, default=0, help='choose id of gpu to use')
        self._parser.add_argument('--classes', type=int, default=3, choices=[3, 1], help='number of classes')
        self._parser.add_argument('--model_path', type=str, default='./model.pth')
        self._parser.add_argument('--num_workers', type=int, default=4)
        
        ### Training configuration.
        self._parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'bce_dice', 'dice'])
        self._parser.add_argument('--weight', type=str, default='', help='weight on 3 classes')
        self._parser.add_argument('--image_dir', action='append', default=['path/to/images'])
        self._parser.add_argument('--label_dir', action='append', default=['path/to/labels'])
        self._parser.add_argument('--train_val_split', type=str, default='', help='[num of partitions, idx of partition for val]')
        self._parser.add_argument('--train_batch_size', type=int, default=32, help='batch size for training')
        self._parser.add_argument('--val_batch_size', type=int, default=32, help='batch size for validation')
        self._parser.add_argument('--accumulation_steps', type=int, default=32, help='number of steps to update params')
        self._parser.add_argument('--lr', type=float, default=0.0001)
        self._parser.add_argument('--num_epochs', type=int, default=30)
        self._parser.add_argument('--save', type=str, default='best', choices=['best', 'all'])
        self._parser.add_argument('--resume', type=str, default='False', choices=['True', 'False'])
        self._parser.add_argument('--resume_from', type=int, default=0, help='number of epochs to start counting')
        
        self._parser.add_argument('--names_file', type=str, default='', help='csv file containing names list')
        self._parser.add_argument('--test_round', type=int, default=0, help='round of train/test')

        ### Testing configuration.
        self._parser.add_argument('--test_batch_size', type=int, default=32, help='batch size for testing')
        
        ## test on train
        self._parser.add_argument('--test_image_dir', type=str, default='path/to/images')
        self._parser.add_argument('--test_label_dir', type=str, default='path/to/labels')
        self._parser.add_argument('--pred_mask_dir', type=str, default='path/to/pred_masks')
    

    def parse(self):
        cfg = self._parser.parse_args()
        return cfg
