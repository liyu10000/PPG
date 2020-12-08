## Ship Part Segmentation

Overview: Predict the two segmenting lines between TS and BT, and BT and VS. Combining the whole ship mask, we can get the three parts of the ship: TS, BT, and VS.

### Model
 - The model builds upon [HorizonNet](https://github.com/sunset1995/HorizonNet), with some major changes.
 - The basic idea is to predict two lines with the same length as the width of input image. If we regard the boundary between TS/BT or BT/VS as the distance from the curve to image top, the boundary can be represented as a list of distances. The distances are meaningful only when they are inside the ship area. This means that even though we work on a line which spans the entire image, the part to evaluate is those points within the ship. This also motivates us to consider cropping the ship area out of original image and focusing on that.

### Input
 - **Images**: ships are cropped from original images based on whole ship masks, and resized to 640x480.
 - **Labels**: txt files containing the list of distances representing line segments for TS/BT and BT/VS boundaries. Extra information are also included, such as the original size of image, and the position where cropping is performed, so that the cropped ship or predicted parts can be placed back to the original image. Labels are only used at training. As we will see in data preprocessing section, the txt files are still generated at testing phase, because image size and crop position information will be used at postprocessing. The list of distances in this case will be initialized with all zeros and get ignored.

### Output
 - **Labels**: txt files with two predicted list of distances representing the two line segments. Note the length of the list is 640. Have to resize and fit into the original image later on. The txt files have the same format as input txts. Only the distance data will be used.
 - Curves: Blank images with two curves are also generated for visualization. The size of images is 640x480, and it corresponds the cropped ship.

### Training Data
 - Currently we have 140 labeled image/label pairs.

### Data Preprocessing
Data Preprocessing is quite different in making training data and testing data. Generally, making training data contains three major steps whereas making testing data only needs one step.

#### Make Training Data
We first convert the part labels into an uniform format (step 1) and then create images and txt labels for training (step 2&3, check function `segpart/preprocess.py:gen_part_data_wi_labels`).

1. Prepare label files. 

Since label files for part segmentation come in different formats, we prefer an uniform label format to start with. It is the json format that can be opened and edited by an open-source annotation tool [labelme](https://github.com/wkentaro/labelme). The reason we choose to convert to the `labelme` format is that we found the raw labels sometimes have to be modified to fit into current setup of inputs. For example, we expect the curve to cover the entire boundary between TS/BT or BT/VS. In some cases we have to manually extend the curve a little bit.

> For making training data, we assume that if parts are labeled, the whole ship masks are also labeled. Currently what we do is first labeling the whole ship masks using `labelme` and then adding part labels from csvs to the json files created by `labelme`. We wrote a short script to handle this. It is `segpart/preprocess.py:add_csv_to_whole_json`. If you just want to convert csvs files to json format without whole ship masks, you can use the script `segpart/preprocess.py:csv_to_json`. Note that the names for the two lines are now `seg1` and `seg2` for `TS_BT` and `BT_VS` respectively.
>
> For making testing data, no labeled whole ship masks or parts are assumed to exist, but the predicted whole ship masks from previous stage are needed.

2. Generate txt files for training

After we get the json formatted label files, we can generate the txt files for training. This step includes: 
 - find the position of ship using whole ship mask labels
 - crop ship and resize to 640 x 480
 - convert part labels to lists of distances and save to txts

3. Data augmentation

We conduct two sets of data augmentation strategies: 
 - random rotate and shift
 - RGB channel flip and left-right flip

#### Make Testing Data
Making testing data is more straightforward than making training data. Only the original image and the predicted whole ship mask are needed. The whole ship mask is applied to find the position of ship on the original image. The function to process this is `segpart/preprocess.py:gen_part_data_wo_labels`.

### Data Postprocessing
 - The outputs from the model are txt files with two lists of distances representing TS/BT and BT/VS boundaries. The length is now 640, the width of input images. We have to expand the lines, fit them into the original image, and generate image masks with three parts. 


### Hyper-parameters
#### Preprocess configs
 - image_dir: Path to original images. Used for both training and testing.
 - label_dir: Path to part seg labels. Used for training.
 - whole_mask_dir: Path to predicted whole ship masks. Need to be in original size. Used for testing.
 - image_layout_dir: Path to store cropped ship images. Size 640 x 480. Used for both training and testing.
 - label_layout_dir: Path to store txt files. Used for both training and testing.
 - image_layout_aug_dir: Path to store augmented ship images. Size 640 x 480. Used for training.
 - label_layout_aug_dir: Path to store augmented txt files. Used for training.

#### Training configs
 - keyword: The word to name saved checkpoints.
 - ckpt: The dir to store saved checkpoints.
 - pth: Full path to the checkpoint for resuming training. Set to `None` if training from scratch.
 - backbone: The backbone encoder.
 - no_rnn: Whether or not to apply a LSTM to smooth lines.
 - img_dirs: List of paths to images. Usually it is `[image_layout_dir, image_layout_aug_dir]`.
 - txt_dirs: List of paths to txts. Usually it is `[label_layout_dir, label_layout_aug_dir]`.
 - train_val_split: A list with two integers specifying the number of partitions and partition number for validation.
 - num_workers: Number of workers to load data.
 - train_batch_size: Training batch size.
 - val_batch_size: Validation batch size.
 - epochs: Number of epochs.
 - optim: Optimizer.
 - lr: Learning rate.
 - lr_pow: Power in poly to drop learning rate.
 - warmup_lr: Staring learning rate for warm up.
 - warmup_epochs: Number of warm up epochs.
 - save_every: Number of epochs to save a checkpoint.

#### Testing configs
 - calc_loss: Whether or not to calculate loss. Only applicable when txt files contain real part labels.
 - test_img_dir: Path to images. It should be like `image_layout_dir`.
 - test_txt_dir: Path to txts. It should be like `label_layout_dir`.
 - test_pth: Full path to the checkpoint for loading weights.
 - pred_dir: The dir to store predicted txt files.
 - test_num_workers: Number of workers to load data.
 - test_batch_size: Test batch size.

#### Postprocess configs
There are two variables inheriting from above: `label_layout_dir` and `whole_mask_dir`.
 - orisize_pred_dir: The dir to store part seg masks of original sizes.
