## Whole Ship Segmentation

Overview: Apply image segmentation algorithms to detect the ship area in an image.

### Model
 - The model is a UNet based architecture, with ResNet34 as the backbone. 
 - We use an open source package shipped with UNet APIs: [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch). If you check the page, you will find there are other backbones available for UNet, and also other image segmentation models such as FPN and DeepLabV3. Given the fact that segmenting ship from an image doesn't require very detailed feature examination and representation learning, we decide to use the popular and well developed model UNet. ResNet34 should be sufficient to encode image features for ship detection.

### Input and Output
 - The input to the model is images with size 640 x 480.
 - The output of the model is the same to the input: predicted ship masks with size 640 x 480.

### Training Data
 - Currently we manually labeled 726 images, so we have 726 image & label pairs.

### Data Preprocessing
 - All raw images are resized to 640 x 480 before feeding to the UNet model. The ratio of image height to width is determined to represent the majority of aspect ratios of given images. 
 - For training, the ground truth whole ship labels should also be resized to the same as images.

### Data Postprocessing
 - Raw outputs from the model have to be resized to the size of the original image.


### Hyper-parameters

#### Common configs
 - **seed**: The random seed for python, pytorch, and numpy.
 - **gpu**: The id of GPU to run the model.
 - **backbone**: The backbone encoder for the UNet segmentation model.
 - **classes**: The number of classes. In this task it should always be 1.
 - **model_path**: The full path of weights file.
 - **num_workers**: Number of workers in loading data.

#### Training configs
 - **image_dir**: Path to training images, or a list of paths. Should be images of size 640 x 480.
 - **label_dir**: Path to training labels, or a list of paths. Should be whole ship masks of size 640 x 480.
 - **train_val_split**: A list with two integers specifying the number of partitions and partition number for validation.
 - **train_batch_size**: Train batch size.
 - **val_batch_size**: Validation batch size.
 - **accumulation_steps**: Number of training batches to perform back propagation. It is useful when larger batch size is needed while GPU memory doesn't support.
 - **loss**: Loss function.
 - **lr**: Learning rate.
 - **num_epochs**: Number of training epochs.
 - **save**: Save all model weights at each epoch, or only save the best model.
 - **resume**: Boolean value to indicate whether to resume training. If resume training, the model will load weights from **model_path**.
 - **resume_from**: The index of epoch to start counting epochs.

#### Testing configs
 - **test_image_dir**: Path to testing images, or a list of paths. Should be images of size 640 x 480.
 - **test_label_dir**: Path to testing labels, or a list of paths. Can also be `None` or `[]` if without labels.
 - **test_batch_size**: Test batch size.
 - **pred_dir**: The directory to store raw outputs of segmentation model. Note the predicted ship masks are of size 640 x 480.
 - **orisize_mask_dir**: The directory of the images or labels with original sizes. It is used to infer original size of images.
 - **orisize_pred_dir**: The directory to store prediction outputs with original sizes.
