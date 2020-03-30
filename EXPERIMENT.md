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