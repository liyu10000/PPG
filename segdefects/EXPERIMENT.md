## Experiment Log

### Exp1 (04/21/2020)
 - Idea
1. Cut image&label pairs into 224x224 patches and train a segmentation network
2. Run *3-class* segmentation
3. Cut all patches. This could be problematic because it introduces some negative samples.

 - Config
1. Use all 27 high-quality cases at train and test.
2. Use all 71 low-quality cases at train and test.
2. Run 30 epochs.

 - Result
1. Results are not good. Almost all patches are identified as defects.


### Exp2 (04/28/2020)
 - Idea
1. Cut image&label pairs into 224x224 patches and train a segmentation network
2. Run *3-class* segmentation
3. Cut patches with defect labels. This could be problematic it assumes defects on all patches.

 - Config
1. Use all 27 high-quality cases at train and test.
2. Use all 71 low-quality cases at train and test.
2. Run 30 epochs.

 - Result
1. Results are not good. Almost all patches are identified as defects.


### Exp3 (05/05/2020)
 - Idea
1. Cut image&label pairs into 224x224 patches, guided by whole vessel segments
2. Run *1-class* segmentation

 - Config
1. Use high-quality or low-quality cases separately. 
2. Also set aside 5 images and train on remaining high-quality set or low-quality set.
3. Also use half of low-quality set for training and test on 5 high-q images.
3. Run 30 epochs.
4. BCE-DICE loss.

 - Result
1. Naming conventions: high (train on high and test on high), high2low (train on high and test on low), low (train on low and test on low), low2high (train on low and test on high), high2five (train on reduced high and test on five high), low2five (train on low and test five high), lowhalf2five (train on half low and test on five high).


### Exp4 (05/10/2020)
 - Idea
1. Same as *Exp3*.

 - Config
1. Use high-quality and low-quality data together at training.
2. Run 30 epochs.
3. BCE-DICE loss.

 - Result
1. There is obvious improvement on low-quality data, whereas less improvement on high-quality data.


### Exp5 (05/18/2020)
 - Idea
1. Train on high-q and low-q data together, except 5 high-q images as test set.

 - Config
1. Run 30 epochs.
2. BCE-DICE loss.

 - Result
1. Slight improvement on recall of delamination.


### Exp6 (05/19/2020)
 - Idea
1. *3-class* segmentation.
2. Cut patches based on vessel shape.
3. Train on high-q and low-q data together, except 5 high-q images as test set.

 - Config
1. Run 30 epochs. Extend to 90 epochs.
2. BCE-DICE loss.

 - Result
1. Segments at 30 epoches are bad. Almost all patches are identified as defects.
2. Segments at 90 epoches.


### Exp7 (05/20/2020)
 - Idea
1. *1-class* segmentation. 
2. Assign weights to different types of defects.
3. Train on high-q and low-q data combined, except 5 high-q images as test set.

 - Config
1. weight set at [1.0,2.0,1.0], [1.0,4.0,2.0], [2.0,4.0,1.0], [1.0,4.0,1.0].
2. Run 30 epochs.
3. BCE-DICE loss.

 - Result
1. weight 142 looks better.


### Exp8 (05/24/2020)
 - Idea
1. *1-class* segmentation.
2. Train on high-q and low-q, but split train/val sets and alternate three splits. Test on 5 high-q images.
3. Assign weights.

 - Config
1. Run 30/60/90 epochs.
2. Weight set at [1.0,1.0,1.0], [1.0,4.0,2.0].
3. When running weight 142, correct and update labels of 7HR: Use labelme to manually add missing polygon and regenerate masks.

 - Result
1. For weight 111, 90p is only slightly better than 30p, whereas 60p is worse than 30p. It suggests no need to run 90p.
2. Model trained with train/val split data performs comparably to model trained and validated on same data.
3. Weight 142 at 30p yields better results than weight 111.


### Exp9 (05/24/2020)
 - Idea
1. One-fold data augmentation on delamination.
2. *1-class* segmentation.
3. Train on high-q and low-q, but split train/val sets and alternate three splits. Test on 5 high-q images.
4. Assign weights.

 - Config
1. Run 30/60/90 epochs.
2. Weight set at [1.0,1.0,1.0], [0.9,1.0,1.0], [1.0,1.1,1.1].

 - Result
1. Data augmentation leads to significant improvement.
2. For weight 111, 90p model is slightly better than 30p.


### Exp10 (05/25/2020)
 - Idea
1. Semi-supervised training, on augmented data from *Exp9*.
2. *1-class* segmentation.
3. Iteratively update block fouling and update data loader.

 - Config
1. Run 30 epochs.
2. Determine there is block fouling in patches by checking if area of ground truth fouling is over 50%.

 - Result
1. Results are not as good as *Exp9*.


### Exp11 (05/25/2020)
 - Idea
1. Train on low-q data and finetune high-q data. With data augmentation.
2. *1-class* segmentation.

 - Config
1. Run 30 epochs on low-q data, then 30 epochs on high-q data
2. Lr at low-q is 0.0001, lr at high-q is 0.00001.

 - Result
1. Results are only comparable to those without data augmentation.


### Exp12 (06/06/2020)
 - Idea
0. Received 2nd data set.
1. Augment delamination & corrosion on new high-q set. Augment corrosion on new low-q set.
2. Found a fatal error in previous exps: delamination augmentation data overlaps with testing set!

 - Config
1. Manually select 6 samples from new high-q set.
2. Train on 1st & 2nd data and test on 5in1st & 6in2nd set (expA), name prefix: bce_dice. 
3. As a comparison, train on 2nd and test on 6in2nd set (expB), name prefix: bce_dice_on2nd.

 - Result
1. expB is reasonably good, meaning only low-reso data is enough on low-reso images.
2. For expA and expB, 60p is the best.



### Exp13 (06/20/2020)
 - Idea
0. Received 3rd data set.
1. As labels on three types are quite balanced, no augmentation is executed.
3. Changed ways of calculating precision and recall.

 - Config
1. Manually select 6 samples from new high-q set.
2. Train on 1st, 2nd & 3rd data and test on 5in1st, 6in2nd & 6in3rd set, name prefix: bce_dice. 

 - Result
1. For all three sets of test data, 90p is the best. (under new p&r calculating scheme)


### Exp14 (06/20/2020)
 - Idea
1. Work on 1st, 2nd and 3rd dataset. 
2. *3-class* segmentation.

 - Config
1. Same data split as in *Exp13*.

 - Result
1. Results are bad.


### Exp15 (06/21/2020)
 - Idea
1. Finetune patch-level predicted, pieced togethered defect labels, by concatenate it with original image and tune against ground truth labels, on image level.
2. *1-class* finetune segmentation.
3. 1st step: use bce_dice_60p.pth from *Exp12* to predict all patches (training and testing). 
   2nd step: piece together all patches to form complete predicted labels. Areas not covered are set to zero, and overlapping areas are averaged.
   3rd step: prepare images/labels in size 640x480.

 - Config
1. train 220, val 27, test 17.
3. 90 epochs, save every 30 epochs.

 - Result
1. Segmentation results are slightly better than *Exp13*. Improved on recall, but degraded on precision.
2. Results on 3rd set are not good. I guess it's because the initial labels are from *Exp12*, which is not trained on 3rd set. The initial guess label of 3rd set are bad.
3. 30p is the best.


### Exp16 (06/21/2020)
 - Idea
1. Same idea as in *Exp15*, but use bce_dice_30p.pth from *Exp13* to generate initial guess labels.

 - Result
1. Not much different to *Exp13*.


### Exp17 (07/02/2020)
 - Idea
0. 4th set of data arrived.
1. Iteratively run train/test to find out images that fail defect segmentation.
2. BCE-DICE loss

 - Config
1. 10 rounds of train/test splits on first three sets. 11th and 12th rounds for 4th set. 13th round for missing data.
2. Determine images fail in segmentation if F1 score is less than 0.2.


### Exp18 (07/06/2020)
 - Idea
1. Train on all four sets, with/without failing samples. Set aside test data for comparison.
2. BCE-DICE loss

 - Config
1. Num of total images: 347. Num of test: 18. Num of train: 329, or 293 without failing samples.
2. 30 epochs.

 - Result
1. No significant difference between two training strategies.


### Exp19 (07/15/2020)
 - Idea
1. Use smaller backbone: resnet18, resnet34. Use all training samples.
2. BCE-DICE loss

 - Config
1. 30 epochs.

 - Result
1. Resnet18 is obviously worse than xception.
2. Resnet34 is almost the same with xception, only with slight degradation.


### Exp20 (07/16/2020)
 - Idea
1. Got 5th dataset. Use all data in this experiment. Augment on low-reso images.
2. Assign a lower weight to background at training. (1.0, 0.9, 0.5)
3. Slice patches into smaller sizes (128) and separate background patches from defect patches.
4. File naming: w05, w09, w10 are exps on size 224 and all data. 224nobg_w10 is on size 224 and without background patches. 128all_w05, 128all_w09 are exps on size 128 and all data.

 - Config
1. Combined individual sets into a big set.
2. Removed all defect type specific augmentation data.
3. 30 epochs. BCE-DICE. Use resnet34 as backbone.

 - Result
1. For comparison in w05, w09, w10, Weight 1.0 (the default setup of previous experiments) is worst in terms of F1, Weight 0.9 and 0.5 share the same F1 scores, with alternating precision and recall.
2. For comparison in 128 and 224 patch size, 128 doesn't yield the same performance as 224. So stick with 224.
3. For comparison in training on all data and training on defect-only data, defect-only data gets better results.
4. For comparison in w09, w10 on defect-only data, and corresponding all data, w10 gets better results.
5. Conclusion: patch 224, w10, on defect-only data gets best results.


### Exp21 (07/19/2020)
 - Idea
1. Got additional data for 5th set.
2. Follow the best param setup from *Exp20*, but train for 90 epochs in an iterative fashion.

 - Config
1. Train for 90 epochs.

 - Result
1. Checkpoints at 30th, 60th, 90th performs similarly in terms of F1 score. So stick with 30 epochs.


### Exp22 (07/26/2020)
 - Idea
1. Got additional data for 6th set.
2. Follow the best param setup from *Exp20*: patch 224, resnet34, defect-only data.
3. Comparison1: weight 0.9 vs. 1.0 (bce_dice_nobg_w09 vs. bce_dice_nobg_w10).
   Comparison2: original data vs. original + HR-downsampled data (bce_dice_nobg_w10 vs. bce_dice_ds2_nobg_w10).
   Comparison3: original HR+SR data vs. original HR data vs. original HR+SR data without some supres (bce_dice_nobg_w10 vs. bce_dice_hr_nobg_w10 vs. bce_dice_rm_nobg_w10).
   Comparison4: amount of HR+SR data vs. performance.
4. Save proba predictions directly for patches and average probability at jointing.

 - Config
1. Train for 90 epochs.
2. Use batch size 128, instead of 64.

 - Result
1. Comparison 1: Weight 1.0 is better than 0.9. Stick with w1.0.
2. Comparison 2: Training with downsampled data doesn't lead to better results. Don't use.
3. Comparison 3: Training on HR+SR > on HR+SR without some SR >> HR only, in terms of HR+SR combined test and also HR/SR separately.
4. Comparison 4: The more HR+SR data, the better the f1 on HR test set. Effect not as clear on SR test set. Will train on SR separately in next experiment.
4. Saving proba and averaging at jointing yields more smooth segments. Lead to better performance for HR, but worse performance for SR. I guess the reason of degradation on SR is averaging makes less predictions.


### Exp23 (08/12/2020)
 - Idea
0. Got 7th set of data.
1. Select gold test set: 10 from HR and 10 from SR. Retrain whole ship segmentation and defect segmentation.
2. At jointing patch predictions, change threshold from 0.5 to 0.3 & 0.4.
3. Comparison: amount of SR data vs. performance.
4. Augment delamination (G) and corrosion (R) for SR.

 - Config
1. Train for 90 epochs.
2. Use batch size 128, patch size 224, backbone resnet34, defect-only data.
3. Use augmented SR data. (named hrsr_nobg2_w10)

 - Result
1. As expected, lowering the threshold will increase recall, with slight decrease of precision.
2. Additional augment on GR of SR doesn't seem to make a difference, probably because of the limited number of new training data comparing to overwhelmingly large HR training data.


### Exp24 (08/26/2020)
 - Idea
1. Randomly select test set (three shuffles): 30 from HR and 30 from SR. Retrain defect segmentation.
2. Retrain defect segmentation on all labeled data.

 - Config
1. Train for 90 epochs.
2. Use batch size 128, patch size 224, backbone resnet34, defect-only data.

 - Result
1. Different shuffles can lead to different results. 2nd shuffle works best.


### Exp25 (09/08/2020)
 - Idea
0. Got 8th set of data. Of which 31 images are from gold test set.
1. Train defect segmentation on all labeled data (A total of 667 images, of which 3 are unusable).
2. To avoid info leaking, also train on pure labeled data (636 images, of which 3 are unusable, removing 31 in POC).

 - Config
1. Train for 90 epochs.
2. Use batch size 128, patch size 224, backbone resnet34, defect-only data.

 - Result
1. As always, pick epoch-90 for testing.


### Exp26 (10/21/2020)
 - Idea
0. Got 9th set of data. Of which 55 images in gold test test are covered. Total number of images is now 726.
1. To avoid info leaking, train on pure labeled data (671 images, of which 3 are unusable, removing 55 in POC).
2. Also augment on HR images. (note originally only augment SR images)

 - Config
1. Train for 90 epochs.
2. Use batch size 128, patch size 224, backbone resnet34, defect-only data.

 - Result
1. As always, pick epoch-90 for testing.


### Exp27 (12/27/2020)
 - Idea
1. Teacher-Student training scheme.
2. Use 55 POC images as test set. Only aug SR.

 - Config
1. Train teacher:
	- Use with fine, labeled data.
	- Train without low quality (confidence) labels. As a comparison, also with low-q labels.
	- Exclude patches with labeled areas less than 1 percent.
	- 30 epochs. Save model at every epoch.
2. Update labels:
	- Pick a teacher model that performs well but doesn't overfit on labels.
	- Take union of manual labels and predicted labels. As a comparison, also take intersection.
3. Train student:
	- Use updated labels.