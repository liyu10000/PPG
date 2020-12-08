import os
from easydict import EasyDict as edict

cur_path = os.path.dirname(os.path.abspath(__file__))

cfg = edict()
# Users can get config by:
#   from config import cfg


###
# Universal configs
###
cfg.seed = 42
cfg.gpu = 1


###
# For testing the pipeline
###
cfg.test_image_dir = 'datasets/images'

###
# Whole ship segmentation
###
cfg.segwhole = edict()
cfg.segwhole.backbone = 'xception'  # 'xception', 'resnet18', 'resnet34', 'resnet50'
cfg.segwhole.classes = 1
cfg.segwhole.model_path = 'weights/segwhole.pth'
cfg.segwhole.num_workers = 4
# Training configuration
cfg.segwhole.orisize_image_dir = ''
cfg.segwhole.orisize_label_dir = ''
cfg.segwhole.image_dir = ''
cfg.segwhole.label_dir = ''
cfg.segwhole.train_val_split = [9, 0]  # num of partitions, idx of partition for val
cfg.segwhole.train_batch_size = 4
cfg.segwhole.val_batch_size = 4
cfg.segwhole.accumulation_steps = 16  # number of steps to update params
cfg.segwhole.loss = 'bce_dice'  # bce, dice, or bce_dice
cfg.segwhole.lr = 0.0001
cfg.segwhole.num_epochs = 1
cfg.segwhole.save = 'best'  # best or all
cfg.segwhole.resume = False  # True or False
cfg.segwhole.resume_from = 0  # index of epoch to resume training
# Testing configuration
cfg.segwhole.orisize_test_image_dir = cfg.test_image_dir
cfg.segwhole.test_image_dir = 'datasets/images-segwhole'  # [os.path.join(cur_path, 'dataparts/segwhole/images9'),]
cfg.segwhole.test_label_dir = []  # [os.path.join(cur_path, 'dataparts/segwhole/labels9'),]
cfg.segwhole.test_batch_size = 4
cfg.segwhole.pred_dir = 'datasets/labels-segwhole-pred'
cfg.segwhole.orisize_mask_dir = cfg.test_image_dir
cfg.segwhole.orisize_pred_dir = 'datasets/outputs/segwhole/'


###
# Ship part segmentation
###
cfg.segpart = edict()
# Pre-process configuration
cfg.segpart.image_dir = cfg.test_image_dir
cfg.segpart.label_dir = ''
cfg.segpart.whole_mask_dir = cfg.segwhole.orisize_pred_dir
cfg.segpart.image_layout_dir = 'datasets/images-segpart'
cfg.segpart.label_layout_dir = 'datasets/labels-segpart'
cfg.segpart.image_layout_aug_dir = ''
cfg.segpart.label_layout_aug_dir = ''
# Training configuration
cfg.segpart.keyword = 'cnn+rnn'
cfg.segpart.ckpt = ''
cfg.segpart.pth = None  # os.path.join(cfg.segpart.ckpt, cfg.segpart.keyword+'.pth')
cfg.segpart.backbone = 'resnet34'  # 'resnet18', 'resnet34', 'resnet50', 'densenet121'
cfg.segpart.no_rnn = False
cfg.segpart.img_dirs = [cfg.segpart.image_layout_dir, cfg.segpart.image_layout_aug_dir]
cfg.segpart.txt_dirs = [cfg.segpart.label_layout_dir, cfg.segpart.label_layout_aug_dir]
cfg.segpart.train_val_split = [9, 0]
cfg.segpart.num_workers = 4
cfg.segpart.train_batch_size = 4
cfg.segpart.val_batch_size = 4
cfg.segpart.epochs = 30
cfg.segpart.optim = 'Adam'  # Adam or SGD
cfg.segpart.lr = 1e-4
cfg.segpart.lr_pow = 0.9
cfg.segpart.warmup_lr = 1e-6
cfg.segpart.warmup_epochs = 10
cfg.segpart.save_every = 10
# Testing configuration
cfg.segpart.calc_loss = False 
cfg.segpart.test_img_dir = cfg.segpart.image_layout_dir
cfg.segpart.test_txt_dir = cfg.segpart.label_layout_dir
cfg.segpart.test_pth = 'weights/segpart.pth'
cfg.segpart.pred_dir = 'datasets/labels-segpart-pred'
cfg.segpart.test_num_workers = 4
cfg.segpart.test_batch_size = 4
# Post-process configuration
cfg.segpart.orisize_pred_dir = 'datasets/outputs/segpart/'


###
# Defect segmentation
###
cfg.segdefect = edict()
# Pre-process configuration
cfg.segdefect.raw_label_dir = ''
cfg.segdefect.processed_dir = ''
cfg.segdefect.image_dir = cfg.test_image_dir
cfg.segdefect.label_dir = ''
cfg.segdefect.whole_mask_dir = cfg.segwhole.orisize_pred_dir
cfg.segdefect.patch_image_dir = 'datasets/images-segdefect'
cfg.segdefect.patch_label_dir = 'datasets/labels-segdefect'
cfg.segdefect.patch_size = 224
cfg.segdefect.step_size = 112
cfg.segdefect.binary = True  # binary labels (defect vs. no defect), or ternary labels (corrosion, delamination, fouling)
cfg.segdefect.defect_only = True  # only keep patches with defects
# Training configuration
cfg.segdefect.backbone = 'resnet34'  # 'xception', 'resnet18', 'resnet34', 'resnet50'
cfg.segdefect.classes = 1  # 1 or 3
cfg.segdefect.names_file = ''
cfg.segdefect.model_path = ''
cfg.segdefect.train_image_dirs = [cfg.segdefect.patch_image_dir, ]  # can be a single path instead of path list
cfg.segdefect.train_label_dirs = [cfg.segdefect.patch_label_dir, ]  # can be a single path instead of path list
cfg.segdefect.train_val_split = [9, 0]
cfg.segdefect.trainkey = ('train', 1)  # key value pair in names.csv for specifying training samples
cfg.segdefect.takefirst = -1  # number of names in csv file for training
cfg.segdefect.onlySR = False  # only use SR data
cfg.segdefect.loss = 'bce_dice'
cfg.segdefect.weight = 1.0  # weight on no-defect areas
cfg.segdefect.train_batch_size = 16
cfg.segdefect.val_batch_size = 16
cfg.segdefect.accumulation_steps = 16
cfg.segdefect.lr = 0.0001
cfg.segdefect.lr_patience = 3  # number of epochs to reduce lr if no drop in val loss
cfg.segdefect.num_epochs = 1
cfg.segdefect.num_workers = 4
cfg.segdefect.save = 'best'  # best or all
cfg.segdefect.resume = False
cfg.segdefect.resume_from = 0  # id of checkpoint
# Testing configuration
cfg.segdefect.test_backbone = 'resnet34'  # 'xception', 'resnet18', 'resnet34', 'resnet50'
cfg.segdefect.test_classes = 1  # 1 or 3
cfg.segdefect.test_model_path = 'weights/segdefect.pth'
cfg.segdefect.test_num_workers = 4
cfg.segdefect.test_image_dir = cfg.segdefect.patch_image_dir
cfg.segdefect.test_label_dir = None
cfg.segdefect.test_batch_size = 16
cfg.segdefect.pred_type = 'npy'  # npy or png
cfg.segdefect.pred_patch_dir = 'datasets/labels-segdefect-pred'
# Post-process configuration
cfg.segdefect.threshold = 0.3  # threshold to determine if it is defect
cfg.segdefect.orisize_pred_joint_dir = 'datasets/outputs/segdefect/'
# cfg.segdefect.label_dir = None  # ground truth label dir
cfg.segdefect.same_channel = False  # if predicted defect has the same channel to ground truth labels
# cfg.segdefect.whole_mask_dir = None


###
# Defect Classification
###
cfg.clsdefect = edict()
cfg.clsdefect.inp_path = cfg.test_image_dir  # location of input images to be fed to the classification
cfg.clsdefect.out_path = "datasets/outputs/clsdefect/"  # folder location of the output image
cfg.clsdefect.model_path = "weights/clsdefect.pth"  # file path to the model
cfg.clsdefect.test_batch_size = 32  # batch size to work on simultaneously
cfg.clsdefect.test_patch_size = 64  # patch size for defect labeling process
cfg.clsdefect.test_stride_size = 32  # stride size for defect labeling process
cfg.clsdefect.ratio_threshold = 0.1  # area percentage to consider wether to label a patch is defected or not
cfg.clsdefect.thresholds = [0.5, 0.5, 0.24]  # thresholds used to classify as defected (order: corrosion, fouling, delamination)
