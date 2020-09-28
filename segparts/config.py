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
        self._parser.add_argument('--backbone', type=str, default='xception', choices=['xception', 'resnet18', 'resnet34', 'resnet50'])
        self._parser.add_argument('--classes', type=int, default=3, choices=[6, 3, 1], help='number of classes')
        self._parser.add_argument('--whole_mask_dir', action='append', default=['path/to/whole/mask/'])
        self._parser.add_argument('--model_path', type=str, default='./model.pth')
        self._parser.add_argument('--num_workers', type=int, default=4)
        
        ### Training configuration.
        self._parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'bce_dice', 'dice'])
        self._parser.add_argument('--image_dir', action='append', default=['path/to/images'])
        self._parser.add_argument('--label_dir', action='append', default=['path/to/labels'])
        self._parser.add_argument('--train_val_split', type=str, default='', help='[num of partitions, idx of partition for val]')
        self._parser.add_argument('--train_batch_size', type=int, default=16, help='batch size for training')
        self._parser.add_argument('--val_batch_size', type=int, default=16, help='batch size for validation')
        self._parser.add_argument('--accumulation_steps', type=int, default=16, help='number of steps to update params')
        self._parser.add_argument('--lr', type=float, default=0.0001)
        self._parser.add_argument('--num_epochs', type=int, default=30)
        self._parser.add_argument('--save', type=str, default='best', choices=['best', 'all'])
        self._parser.add_argument('--resume', type=str, default='False', choices=['True', 'False'])
        self._parser.add_argument('--resume_from', type=int, default=0, help='number of epochs to start counting')
        
        ### Testing configuration.
        self._parser.add_argument('--test_batch_size', type=int, default=4, help='batch size for testing')
        
        ### Finetune on isPart classification
        self._parser.add_argument('--is_part_csv', type=str, default='path/to/is_part_csv')

        # ## test on train
        # self._parser.add_argument('--test_image_dir', action='append', default=['path/to/images'])
        # self._parser.add_argument('--test_label_dir', action='append', default=['path/to/labels'])
        # self._parser.add_argument('--pred_mask_dir', type=str, default='path/to/pred_masks')
        
        # test on separate testset (images/labels)
        self._parser.add_argument('--test_image_dir', action='append', default=['path/to/test/images'])
        self._parser.add_argument('--test_label_dir', action='append', default=['path/to/test/labels'])
        self._parser.add_argument('--pred_npy_dir', type=str, default='path/to/pred/npy/dir')
        self._parser.add_argument('--pred_img_dir', type=str, default='path/to/pred/img/dir')
        self._parser.add_argument('--orisize_mask_dir', type=str, default='path/to/original/label/dir')
        self._parser.add_argument('--orisize_save_dir', type=str, default='path/to/save/original/sized/pred/masks')


    def parse(self):
        cfg = self._parser.parse_args()
        return cfg
