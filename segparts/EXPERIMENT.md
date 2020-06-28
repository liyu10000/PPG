## Experiment Log

### Exp1 (03/04/2020)
 - Idea
1. Added more labled data from Steve.
2. Run *6-class* segmentation

 - Config
1. Validate on 12 images, train on the rest (96). 
2. Do three rounds of train-val CV. Final # epochs is 90.

 - Result
1. Result on test (first 12) are saved in `xcep_tv_90th`
2. Result on external test set are save in `Hi-Resx2`


### Exp2 (03/04/2020)
 - Idea
1. Dataset are the same as in *Exp1*
2. Run *3-class* segmentation

 - Config
1. Validate on 12 images, train on the rest (96). 
2. Do three rounds of train-val CV. Final # epochs is 90.

 - Result
1. Result on test (first 12) are saved in `xcep_tv_90th`
2. Result on external test set are save in `Hi-Resx2`
3. Result of 3-class segmentation on external dataset is worse than that of Exp1.


### Exp3 (03/29/2020)
 - Idea
1. Added additional dataset
2. Run *3-class* segmentation
3. Tried bce, dice and bce_dice loss, respectively.
4. Also tried resizing image without padding.

 - Config
1. Validate on 12 images, train on the rest (92).
2. Do three rounds of train-val CV. Final # epochs is 90.
3. Set lr to 0.0001.

 - Result
1. Bce_dice is slightly better than bce.
2. Dice is the worst.
3. Resizing without padding is not good.


### Exp4 (04/09/2020)
 - Idea
1. Manually horizontal-flip images and labels.
2. Manually JPEG compress training data, to accommodate low-res testing data. Use multiple strengths of compression.
3. All data are resized without padding.

 - Config
1. Total number of images is 685. Validate on 70 images, train on the rest (615).
2. One round of train-val is enough. Final # epochs is 30.

 - Result
1. Results are reasonably good, though not as good as the one from *Exp3*.


### Exp5 (04/26/2020)
 - Idea
1. Manually labeled the whole ship, stored in *labels_whole* directory.
2. Augment data by varied JPEG compressions.
3. One-class segmentation.
4. All data are resized without padding.
5. BCE-DICE loss.

 - Config
1. Total number of images is 685. Validate on 70 images, train on the rest (615).
2. One round of train-val is enough. Final # epochs is 30.

 - Result
1. Results are very good. Will use it as the base mask to work on.


### Exp6 (04/27/2020)
 - Idea
1. Manually labeled three parts on myself, stored in *labels_new* directory, new training labels in *labels_3cls_new* directory.
2. Augment data.
3. Three-class segmentation.
4. Without padding.
5. BCE-DICE loss.

 - Config
1. Total number of images is 685. Validate on 70 images, train on the rest (615).
2. One round of train-val is enough. Final # epochs is 30.

 - Result
1. Results are really bad. Something must be wrong.


### Exp7 (06/16/2020)
 - Idea
1. Whole ship segmentation.
2. New data from set2 and set3 of defect segmentation. Combine with data in *Exp5* (add 6 more without clear TS/BT/VS parts).
3. One-class segmentation.
4. BCE-DICE loss.

 - Config
1. Total number of images is 399. Set aside 15 from set3 as test set. For old data, only use jpg3040 compressions.
2. Train 90 epochs, save every 30 epochs.

 - Result
1. Test result on 'Segmentation_Test_Set' and 'set3' sets are good.
2. 60p seems to be best.


### Exp8 (06/27/2020)
 - Idea
1. Use whole ship mask as 4th channel at training part segmentation.
2. *3-class* segmentation.

 - Config
1. Use original images for part segmentation, together with their whole ship masks.
2. Also use aug_3040 images.
3. Try BCE and BCE-DICE, and BCE-DICE (plain, without random contrast and random brightness aug, because it would slightly change whole ship mask).

 - Result
1. BCE seems to be best on 1st set, BCE-DICE seems to be best on 2nd set. 
2. BCE-DICE produces excess predictions outside of ship, but can get better by only considering ship regions.
3. Plain BCE-DICE reduced excess predictions, but seemed to yield a lower performance.
