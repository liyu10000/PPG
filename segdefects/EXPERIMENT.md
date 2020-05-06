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
2. Train and test on all data.
3. Run 30 epochs.

 - Result
