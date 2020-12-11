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


### Exp9 (07/27/2020)
 - Idea
1. Whole ship segmentation. 
2. Use all six sets of data. Do not do jpeg augmentation.

 - Config
1. Total number of images is 522. Use the same test (52 images) as in defect segmentation.
2. Train 90 epochs, save every 30 epochs.


### Exp10 (08/12/2020)
 - Idea
1. Whole ship segmentation. 
2. Use all seven sets of data. Do not do jpeg augmentation.
3. Also train with separate test set (goldtest), use seg results (image 56 and 57) from it.

 - Config
1. Total number of images is 597. Use the same test (20 images) as in defect segmentation.
2. Train 90 epochs, save every 30 epochs.
3. 90 epochs are good.


### Exp11 (07/12/2020)
 - Idea
0. HorizonNet first run.
1. Run given scripts on given dataset to make sure scripts are runnable.

 - Config
1. Resnet50 + RNN


### Exp12 (09/21/2020)
 - Idea
0. HorizonNet.
1. Train and predict part boundaries using HorizonNet.
2. Only use foreground ships. Create by applying whole ship masks.
3. Hard augmentation: rgb channel flip and left-right flip.

 - Config
1. 90 cropped and resized images with two boundaries (TS/BT, BT/VS).
2. Images are shaped 640x480 and labels are two 1d vectors (2x640).
3. Resnet34. 90 epochs.
4. Train under three configurations: without mask, with mask on original boundaries, with mask on extended boundaries.

 - Result
1. There is no significant difference between the three configs.


### Exp13 (09/27/2020)
 - Idea
0. HorizonNet.
1. More active augmentation: random rotate and random shift.

 - Config
1. 90 + 351 (rs aug, should be 360, some are neglected) + 441 (c aug, should be 450).
2. Resnet34, and Resnet34+RNN. 90 epochs.
3. Two configs: without mask, with mask on original boundaries.

 - Result
1. Seg results are much better than that in *Exp12*.
2. Train with RNN will smooth the output. The performance however are not really improved.
3. Results are better than segmentation results from UNet, on POC data.


### Exp14 (09/28/2020)
 - Idea
1. Part segmentation with aggressive data augmentation. Serves as baseline to HorizonNet.
2. Crop ship from images. Zero out backgrounds.
3. Use active data augmentation such as random rotate, random shift, and RGB channel flip.

 - Config
1. Total number of training images goes to 104 * 5 * 2 = 1040.
2. Train/val batch size 16. Epochs 90.

 - Result
1. Results are no better than *Exp13*.


### Exp15 (10/25/2020)
 - Idea
0. Received a new batch of part seg data, with 15 images.
1. HorizonNet.
2. More active augmentation: random rotate and random shift.
3. Corrected a bug in code: should pad left/right for the same side. Also tested no padding.

 - Config
0. Currently only consider images with two line segs.
1. 90 + 351 (rs aug) + 441 (c aug). Plus 14 + 56 (rs aug) + 70 (c aug).
2. Resnet34, and Resnet34+RNN. 90 epochs.
3. One config: without mask.

 - Result
1. Under old padding scheme, adding new data leads to minor improvement.
2. Without padding could improve segmentation a little bit. 
3. Correct padding leads to further improvement.


### Exp16 (11/29/2020)
 - Idea
0. Received a new batch of part seg data, with 216 images.
1. HozizonNet.
2. Same data aug as *exp15*. 
3. Corrected one more padding bug.

 - Config
0. Only consider images with two line segs.
1. New data overlaps with POC. Set aside those overlaps (15 images) as test data.

 - Result
1. With additional data, performance on both separate test set and POC set get improved.
2. Remove last block doesn't work.


### Exp17 (11/30/2020)
 - Idea
1. Train our own model, by replace interpolation with fully connected layer.
2. Same data as *Exp16*.

 - Config
1. Train for 180 epochs, with each 60 epochs switching train/val data.

 - Result
1. Training with more data improves result.


### Exp18 (12/10/2020)
 - Idea
1. Train UNet on combined data, without aug. Use as UNet baseline.
2. Set aside 25 images randomly selected from set1 and set2 and use as test set. 
3. The 10 separate test set will also be kept and tested. 

 - Config
1. Train for 90 epochs, with 30 epochs a round.

 - Result
