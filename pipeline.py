import os
import random
import numpy as np
import torch
import warnings

from config import cfg
from segwhole.preprocess import resize
from segwhole.train import Trainer as SegWholeTrainer
from segwhole.test import Tester as SegWholeTester
from segpart.preprocess import gen_part_data_wi_labels, gen_part_data_wo_labels
from segpart.solver import segpart_train, segpart_test
from segpart.postprocess import gen_mask_from_lines
from segdefect.preprocess import csv_to_masks, gen_defect_data
from segdefect.train import Trainer as SegDefectTrainer
from segdefect.test import Tester as SegDefectTester
from segdefect.postprocess import joint_patch, evaluate
from clsdefect.test import main as ClsDefectTester


warnings.filterwarnings("ignore")
print(cfg)
random.seed(cfg.seed)
os.environ["PYTHONHASHSEED"] = str(cfg.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
np.random.seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


### whole ship segmentation
print('===> Starting Whole Ship Segmentation ...')
segwhole_cfg = cfg.segwhole
# segwhole_trainer = SegWholeTrainer(segwhole_cfg)
# segwhole_trainer.start()
resize(segwhole_cfg.orisize_test_image_dir, segwhole_cfg.test_image_dir)
segwhole_tester = SegWholeTester(segwhole_cfg)
segwhole_tester.start()
print('===> Whole Ship Segmentation Finished.')


### ship part segmentation
print('===> Starting Ship Part Segmentation ...')
segpart_cfg = cfg.segpart
# # generate training data
# gen_part_data_wi_labels(segpart_cfg.image_dir, segpart_cfg.label_dir, 
#                         segpart_cfg.image_layout_dir, segpart_cfg.label_layout_dir, 
#                         segpart_cfg.image_layout_aug_dir, segpart_cfg.label_layout_aug_dir)
# generate testing data
gen_part_data_wo_labels(segpart_cfg.image_dir, segpart_cfg.whole_mask_dir, 
                        segpart_cfg.image_layout_dir, segpart_cfg.label_layout_dir)
# segpart_train(segpart_cfg)
segpart_test(segpart_cfg)
# generate part seg masks
gen_mask_from_lines(segpart_cfg.label_layout_dir, segpart_cfg.pred_dir, segpart_cfg.orisize_pred_dir)
print('===> Ship Part Segmentation Finished.')


### defect segmentation
print('===> Starting Defect Segmentation ...')
segdefect_cfg = cfg.segdefect
# # generate training data
# csv_to_masks(segdefect_cfg.raw_label_dir, segdefect_cfg.processed_dir)
# gen_defect_data(segdefect_cfg.image_dir, segdefect_cfg.label_dir, 
#                 segdefect_cfg.patch_image_dir, segdefect_cfg.patch_label_dir, 
#                 segdefect_cfg.step_size, segdefect_cfg.patch_size, 
#                 segdefect_cfg.defect_only,
#                 segdefect_cfg.binary, 
#                 segdefect_cfg.whole_mask_dir)

# generate testing data
gen_defect_data(segdefect_cfg.image_dir, None, 
                segdefect_cfg.patch_image_dir, None, 
                segdefect_cfg.step_size, segdefect_cfg.patch_size, 
                False,
                segdefect_cfg.binary, 
                segdefect_cfg.whole_mask_dir)

# segdefect_trainer = SegDefectTrainer(segdefect_cfg)
# segdefect_trainer.start()
segdefect_tester = SegDefectTester(segdefect_cfg)
segdefect_tester.start()

# post process
joint_patch(segdefect_cfg.pred_patch_dir, segdefect_cfg.orisize_pred_joint_dir, segdefect_cfg.pred_type)
# evaluate(segdefect_cfg.label_dir, segdefect_cfg.orisize_pred_dir,
#          segdefect_cfg.same_channel, segdefect_cfg.whole_mask_dir)
print('===> Defect Segmentation Finished.')


### defect classification
print('===> Starting Defect Classification ...')
ClsDefectTester()
print('===> Defect Classification Finished.')
