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

