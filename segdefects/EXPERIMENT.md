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

