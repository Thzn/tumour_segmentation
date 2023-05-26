This repository contains the code segments for the experiments conducted in `Few-Shot Learning for Medical Image Segmentation: A Review And Comparative Study`. 
We provide specific implementation steps we followed for training SSL-ALP-Net [1] on the Kits-23 Tumour Segmentation dataset. We encourage users to use our pre-processing/modal
training codes for their experimentation purposes. 

## Our experimental results:
<img width="929" alt="Screenshot 2023-05-24 at 5 28 23 pm" src="https://github.com/Thzn/tumour_segmentation/assets/19911856/80139950-f0f5-45a0-a8ac-b235e232dcb4">

## How to run the code:

This repository includes scripts for training a one-way one-shot (few-shot) learning for identifying tumor regions from kidney CT images obtained from the Kits-2023 challenge. As the baseline model, we use the ALP-Net from [1]. 
1. Download the dataset from [Kits-2023](https://kits-challenge.org/kits23/) challenge website. 
2. Use the preprocessing.py code to extract features. 
3. Run the training.py to train the model on natural images
4. Run the training_super.py to train the model on superpixel data in a Self-Supervised manner (i.e. SSL-ALP-Net)
5. Compare the results using comparisons_tum.py script.

## Dependencies:

[1] Ouyang, Cheng, et al. "Self-supervised learning for few-shot medical image segmentation." IEEE Transactions on Medical Imaging 41.7 (2022): 1837-1848.

