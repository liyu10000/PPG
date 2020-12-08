## Defect Segmentation
Overview: Apply image segmentation algorithms to detect defects.

### Model
 - The model is a UNet based architecture, with ResNet34 as the backbone. As demonstrated earlier, defects are more like local features rather than global features, so a deeper neutral network such as ResNet50 or DenseNet121 is not necessary.
 - Similar to whole ship segmentation, we use the open source package with UNet API: [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch).

### Input and Output
 - The original, big images are sliced into small patches of size 224 x 224 for training and testing. 
 - Accordingly, the labels are also sliced into patches for training.

### Training Data
 - We have 726 image & label pairs.

### Data Proprocessing
 - All images are sliced into patches with patch size 224 and step size 112. Random jittering at slice point is also applied to increase flexibility and diversity.
 - For training, only image&label patches with defects are kept. We have observed the areas get labeled tend to be more reliable than areas get ignored. In other words, the unlabeled areas could be defects, whereas it is unlikely that labeled areas are actually not defects.
 - For testing, all image and possibly label patches are kept for testing.

### Data Postprocessing
 - We have to merge the patches back to a single image. Based on the fact that predictions from model are probabilities of pixels being defect or not, we can follow the two steps to generate the final defect labels:
     - Average probabilities if one pixel get multiple predictions. It could happen because our slicing strategy allows overlap.
     - Set a threshold to determine if a pixel is defect or not. Usually it is set at 0.5, but we have found that setting it at a lower value (eg. 0.3) will increase the amount of predictions, which is closer to the ground truth.

### Hyper-parameters
#### Preprocess configs
 - raw_label_dir: the dir containing the labels in csv formats.
 - processed_dir: the dir containing labels in png formats.