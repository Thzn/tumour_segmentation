import glob
import numpy as np
import os
import re
import cv2
import pickle
import matplotlib.pyplot as plt

import time
import skimage.segmentation as seg 

import matplotlib.pyplot as plt
import copy
import skimage

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import label 
import scipy.ndimage.morphology as snm
from skimage import io

# thresholding the intensity values to get a binary mask of the patient
def fg_mask2d(img_2d, thresh): # change this by your need
    mask_map = np.float32(img_2d > thresh)
    
    def getLargestCC(segmentation): # largest connected components
        labels = label(segmentation)
        assert( labels.max() != 0 ) # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC
    if mask_map.max() < 0.999:
        return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = snm.binary_fill_holes(post_mask)
    return fill_mask

# remove superpixels within the empty regions
def superpix_masking(raw_seg2d, mask2d):
    raw_seg2d = np.int32(raw_seg2d)
    lbvs = np.unique(raw_seg2d)
    max_lb = lbvs.max()
    raw_seg2d[raw_seg2d == 0] = max_lb + 1
    lbvs = list(lbvs)
    lbvs.append( max_lb )
    raw_seg2d = raw_seg2d * mask2d
    lb_new = 1
    out_seg2d = np.zeros(raw_seg2d.shape)
    for lbv in lbvs:
        if lbv == 0:
            continue
        else:
            out_seg2d[raw_seg2d == lbv] = lb_new
            lb_new += 1
    
    return out_seg2d



fg_thresh=-500

def make_dataset_nnunet():
        """
        Read images into memory and store them in 2D
        Build tables for the position of an individual 2D slice in the entire dataset
        """
        ###CHAOS Dataset
        #data_root = "/mnt/hdd/sda/ygeo/data/CHAOST2/nnunet_preprocessing/transformed/plan2D_stage0/"
        #img_modality = "MR"
        
        ###SABS Dataset
        #data_root = "/mnt/hdd/sda/ygeo/data/MultiAtlas_CT_Labelling/Abdomen/Abdomen/nnunet_preprocessing/transformed/plan2D_stage0/"
        #img_modality = "CT"
        
        ###KiTS Dataset
        data_root1 = "/mnt/hdd/sda/ygeo/data/Untitled Folder/kits21/kits21/data/preprocessed/transformed/plan2D_stage0/"
        data_root = "/mnt/hdd/sda/ygeo/code/kits23/kits23_project/dataset_converted/nnUNet_cropped_data/Task100_KiTS23/"

        img_modality = "CT"
        c = 0 
        for fid in glob.glob(data_root + "/*.npz")[85:100]:
            c += 1
            data = np.load(fid)['data']
            data = np.transpose(data, (0,3, 2, 1))
            data = data[:,:,::-1,:]

            scan_id=fid.split('/')[-1].split('.')[0]
            for i in range(data.shape[1]):
                msk = data[1,i, ...]
                msk[msk<=0] = 0
                if np.sum(msk) > 0 : ## remove slices with background only
                    
                    with open(f'{data_root1}2d_kits23/image_{scan_id}-z{i}.pkl', "wb") as f:
                        pickle.dump(data[0, i,...], f)


                        pickle.dump(msk, f)

                        res_s = skimage.segmentation.felzenszwalb(data[0,i, ...], min_size = 400, sigma = 1)
                        fgm = fg_mask2d(data[0,i, ...], fg_thresh )
                        super_seg = superpix_masking(res_s, fgm)

                        pickle.dump(super_seg, f)
                        pickle.dump(fgm, f)
    
            print(c, " is processed")
                    
make_dataset_nnunet()
'''
i=249
scan_id='0150'
#data_root = "/mnt/hdd/sda/ygeo/data/CHAOST2/nnunet_preprocessing/transformed/plan2D_stage0/"
data_root = "/mnt/hdd/sda/ygeo/data/MultiAtlas_CT_Labelling/Abdomen/Abdomen/nnunet_preprocessing/transformed/plan2D_stage0/"
data_root = "/mnt/hdd/sda/ygeo/data/Untitled Folder/kits21/kits21/data/preprocessed/transformed/plan2D_stage0/"
        
with open(f'{data_root}2d_kits32/image_{scan_id}-z{i}.pkl', "rb") as f:
    
    I1 = pickle.load(f)
    I2 = pickle.load(f)
    
print(I1.min(), I1.max(),I2.min(), I2.max())

import matplotlib.pyplot as plt
plt.imshow(I1), plt.show()
plt.imshow(I2), plt.show()
'''

