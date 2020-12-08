# Defect classification

## Table of Contents:
1. [Folder Contents](#contents).
2. [Installing Packages](#setup).
2. [Training](#train).
3. [Inference (Testing)](#test).


<a name="contents"></a>

## Folder Contents
1. 'requirements.txt'
contains the rquired packages that need to be installed. It's preferred to use conda with this file.

2. 'checkpoints' 
contains the saved model file 'model.pth'.

3. 'Datasets'
contains different dataset loaders that are either used with training or inference.

4. 'losses'
contains the loss function file that is used in training.

5. 'models'
contains the model file that is used in training/inference.

6. 'preprocessing'
conatians a file used to preprocess the training dataset given by PPG labelers. The objective is to extract patches and a pkl file for each patch that has information about different defects in that patch (in addition to a mask image of these defects)

7. 'logger.py' 
contains a class used in logging information while trianing

8. 'train.py'
is the file used for training. 
For help, type at the parent directory:
> $ export PYTHONPATH=$PYTHONPATH:./
> $ python clsdefect/train.py --help

9. 'test.py'
is the file used for inference. It is expected that part segmentation/whole ship segmentation/defect segmentation outputs are produced already.
For help, type:
> $ python test.py --help

10. 'transforms_image_mask_label.py'
contains some preprocessing classes used in training the model.

11. 'utils.py'
contains auxilary functions that are used all over the code.



<a name="setup"></a>

## Installing Packages
The file 'requirements.txt' contains the packages that are used with this code. To create a conda virtual environment using that file:
>$ conda create --name <name_of_the_environment> --file requirements.txt -c pytorch



<a name="train"></a>

## Training
To train the network from scratch you need to preprocess the dataset first using 'preprocessing/preprocessing_training.py'. The training model expects the input images to be given as list of folders where each of these folders have one or more sample. A sample consists of 3 files that have the same name but ends with '_mask.png', '_meta.pkl', '_patch.png'.

### Preprocessing:
Run:

>$ export PYTHONPATH=./
>
>$ python preprocessing/preprocessing_training.py

to set the python path first and then to generate these samples for a given folder. You can change the location of source data and destination data at the end of this file named 'src_root' and 'dst_root'.

### Train the model:
After preprocessing the data, you can train the model using 'train.py' file.
Use the argument '--datasets' to set the list of locations where the preprocessed data exists (you can also change its default value in the file).
>$ python train.py


<a name="test"></a>

## Inference (Test) 
To test using a pretrained model use 'test.py' file.
You should have the following outputs ready before running this file:
1. Whole Ship Segmentation.
2. Ship Parts Segmentation.
3. Defected Regions Segmentation.

Each output should be contained in a specific folder, you can set these folders using the arguments '--ship_out_path', '--part_out_path' and '--seg_out_path' respectively.
The input images are in the foler set by the argument '--inp_path'
and the output defects mask image are written in the path set by the argument '--out_path'

### Output Folders:
In output path, there are 7 folders:
1. ***comparisons:*** has comparison image files between whole ship segmentation, part segmentation, defect segmentation, classification without suppressing fouling in TS, and classification with suppression of fouling at TS.

2. ***Defect Segm Output:*** has image files of the output of defect segmentation.

3. ***Part Segm Output:*** has image files of the output of ship parts segmentation (TS in Blue, BT in Green and VS in Red).

4. ***Prediction (thresholded):*** has image files of the output of classification without suppressing fouling in TS (Corrosion in Red, Fouling in Green, and Delamination in Blue).

5. ***Predictions (employing Part Segm Output)*** has image files of the output of classification with the suppression of fouling in TS (Corrosion in Red, Fouling in Green, and Delamination in Blue).

6. ***Whole Ship Segm:*** has image files of the output of whole ship segmentation.

7. ***percentages:*** has '.pkl' and '.csv' files for each infered image.

> pd.read_csv(path, index_col=0)

