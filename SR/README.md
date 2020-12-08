# Super Resolution

## Overview
This folder contains the code for training and testing the super resolution part of the pipeline.
The objective is to increase the resolution of input images by factors x2, x3 and x4

## Model
* This code is based on the [EDSR repositroy in github](https://github.com/thstkdgus35/EDSR-PyTorch) with some modifications.
* Examples of possible runs can be found in `src/commands.sh` and `src/demo.sh`.

## Inference
* We have a pretrained model that can be used directly at `src/pretrained/MDSR_PPG_usingDataset1+2.pt`
* This model was trained over the Low Resolution (LR)/ High Resolution (HR) dataset pairs of marine vessels provided by PPG.
* To use this model for inference use the following command inside the `SR/src/` directory:
>$ cd SR/src/\
>$ python main.py --data_test Demo --scale 2+3+4 --pre_train pretrained/MDSR_PPG_usingDataset1+2.pt --test_only --save_results --dir_demo ../../datasets/SR/low-res/
* This will use a pretrained file at `src/pretrained/MDSR_PPG_usingDataset1+2.pt` and read LR images from `../../datasets/SR/low-res/` and save the results at `SR/experiment/test/results-Demo/`
* The names of the files will be kept the same but with adding the prefix `_xN_SR` where `N` is the upsampling scale factor.
 
## Training
* To train the model you have to run the following command inside `SR/src/` directory
>$ python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR_2+3+4 --reset --save_models --pre_train pretrained/MDSR.pt
* This will train the network starting from a pretrained model located at `src/pretrained/MDSR.pt` 
and using an architecture of 80 residual blocks.
* You can change the template to `EDSR` but we found that `MDSR` is good since it provides multiple scales simultaneously and it is smaller.
* The trained model is saved at `src/pretrained/MDSR_2+3+4.pt`
* Default location for the dataset is at `'../../datasets/SR/trainset/` i.e. a directory named `trainset` inside `datasets/SR/` at the parent directory. This path can be changed using the argument 
`--dir_data`. Inside that directory, we expect two directories `HR` and `SR` that contain the high resolution and low resolution images. `SR` directory has 3 subdirectories for different scales `X2`, `X3` and `X4`.   
* You can use the MATLAB file located at `preprocessing/get_LRs_bicubic.m` to generate LR images used in training.

## Changing Configurations
* All default values of the parameters and the configurations can be found in `src/option.py` with their appropriate help.

