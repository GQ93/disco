#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:42:30 2025

@author: lesliemeng
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio
import imagecodecs
import matplotlib.pyplot as plt
from csbdeep.utils import normalize
from skimage import io, filters, measure, morphology, color
from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation,binary_erosion
import pickle

# %% IoU function

def calculate_iou(binary_image, label_image):
    # Ensure the ground truth is boolean
    label_image_bool = label_image.astype(bool)
    binary_image_bool = binary_image.astype(bool)
    
    # True positives (Intersection)
    intersection = np.logical_and(binary_image_bool, label_image_bool)
    
    # Union
    union = np.logical_or(binary_image_bool, label_image_bool)
    
    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)
    
    return iou


# %% Parameters: Input
baseDir = '/Users/lesliemeng/ImPartial2'
cellMkr = 'PECAM'
cellMkr_img = 'pecam'
testFig = "two"
threshold_mkr = 0.75
threshold_nuc = 0.2
sampleID = '15972'
train_idx = 1



# %% True label
#  image0
image_plot0_0 = imageio.imread(f'{baseDir}/Data/{cellMkr}/Snap-{sampleID}.tiff_files/{cellMkr_img}_sc_chan0_0.tif')
plt.imshow(image_plot0_0)
image0_sc_ch0_0 =  np.zeros_like(image_plot0_0[...,0])
image0_sc_ch0_0[image_plot0_0[...,0]>0] = 1
image0_sc_ch0_0 = image0_sc_ch0_0.astype(float)
image0_sc_ch0_0 = binary_closing(image0_sc_ch0_0)
image0_sc_ch0_0_label = binary_fill_holes(image0_sc_ch0_0)
image0_sc_ch0_0_label = np.where(image0_sc_ch0_0_label, 1, 0)
plt.imshow(image0_sc_ch0_0_label)

image_plot1_0 = imageio.imread(f'{baseDir}/Data/{cellMkr}/Snap-{sampleID}.tiff_files/{cellMkr_img}_sc_chan1_0.tif')
plt.imshow(image_plot1_0)
image0_sc_ch1_0 =  np.zeros_like(image_plot1_0[...,1]) 
image0_sc_ch1_0[image_plot1_0[...,1]>0] = 1
image0_sc_ch1_0 = image0_sc_ch1_0.astype(float)
image0_sc_ch1_0 = binary_closing(image0_sc_ch1_0)
image0_sc_ch1_0_label = binary_fill_holes(image0_sc_ch1_0)
plt.imshow(image0_sc_ch1_0_label)

image_plot2_0 = imageio.imread(f'{baseDir}/Data/{cellMkr}/Snap-{sampleID}.tiff_files/{cellMkr_img}_sc_chan2_0.tif')
plt.imshow(image_plot2_0)
image0_sc_ch2_0 =  np.zeros_like(image_plot2_0[...,1]) 
image0_sc_ch2_0[image_plot2_0[...,1]>0] = 1
image0_sc_ch2_0 = image0_sc_ch2_0.astype(float)
image0_sc_ch2_0 = binary_closing(image0_sc_ch2_0)
image0_sc_ch2_0_label = binary_fill_holes(image0_sc_ch2_0)
plt.imshow(image0_sc_ch2_0_label)

# nuclei true label union
image0_sc_nuc = image0_sc_ch1_0_label+image0_sc_ch2_0_label
plt.imshow(image0_sc_nuc)

# nuclei original union
image_plot_nuc = image_plot2_0+image_plot1_0
plt.imshow(image_plot_nuc)

plt.imshow(image_plot0_0)
plt.axis('off')  # Turn off axis numbers and ticks
# Save the plot to a PDF file
plt.savefig(f'/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/Figure/seg/{cellMkr}/{cellMkr}_{sampleID}_mkr_g.pdf', 
            bbox_inches='tight', pad_inches=0, format='pdf',dpi=300)
# Close the plot to free up memory
plt.close()

plt.imshow(image_plot_nuc)
plt.axis('off')  # Turn off axis numbers and ticks
plt.savefig(f'/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/Figure/seg/{cellMkr}/{cellMkr}_{sampleID}_nuc_g.pdf', 
            bbox_inches='tight', pad_inches=0, format='pdf',dpi=300)
plt.close()


# %% Predicted Label. 
## Evaluation 
file_path = f'/Volumes/T9/{cellMkr}/{testFig}/test_results/image{train_idx}.pkl'
with open(file_path, 'rb') as file:
    output = pickle.load(file)

image_gray = output['marker']
image_blurred = filters.gaussian(image_gray, sigma=1)
binary_image = image_blurred > threshold_mkr
plt.imshow(binary_image)

mkr_iou = calculate_iou(image0_sc_ch0_0_label, binary_image)
f"{mkr_iou:.3f}"

# nuclei concatenate
image_gray = output['nuc_in']
image_blurred = filters.gaussian(image_gray, sigma=1)
binary_in = image_blurred > threshold_mkr
plt.imshow(binary_in)

image_gray = output['nuc_ou']
image_blurred = filters.gaussian(image_gray, sigma=1)
binary_out = image_blurred > threshold_nuc
plt.imshow(binary_out)

binary_nuc = binary_in+binary_out
plt.imshow(binary_nuc)
nuc_iou = calculate_iou(image0_sc_nuc, binary_nuc)
f"{nuc_iou:.3f}"


plt.imshow(binary_image)
plt.axis('off') 
plt.savefig(f'/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/Figure/seg/{cellMkr}/{cellMkr}_{sampleID}_mkr_p_{mkr_iou:.3f}.pdf', 
            bbox_inches='tight', pad_inches=0, format='pdf',dpi=300)
plt.close()

plt.imshow(binary_nuc)
plt.axis('off') 
plt.savefig(f'/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/Figure/seg/{cellMkr}/{cellMkr}_{sampleID}_nuc_p_{nuc_iou:.3f}.pdf', 
            bbox_inches='tight', pad_inches=0, format='pdf',dpi=300)
plt.close()


