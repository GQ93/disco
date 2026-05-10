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
import pandas as pd
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


# %% loop
baseDir = '/Users/lesliemeng/ImPartial2'

cellMkr = 'PECAM'
cellMkr_img = 'pecam'

sampleID_set = ['15591','15968','15969','15970','15971','15972','15973']
train_idx_set=[0,1,2,3,4,5,6]

for sampleID, train_idx in zip(sampleID_set, train_idx_set):
    # Process each channel image and compute binary labels
    image_plot0_0 = imageio.imread(f'{baseDir}/Data/{cellMkr}/Snap-{sampleID}.tiff_files/{cellMkr_img}_sc_chan0_0.tif')
    image0_sc_ch0_0 =  np.zeros_like(image_plot0_0[...,0])
    image0_sc_ch0_0[image_plot0_0[...,0]>0] = 1
    image0_sc_ch0_0 = image0_sc_ch0_0.astype(float)
    image0_sc_ch0_0 = binary_closing(image0_sc_ch0_0)
    image0_sc_ch0_0_label = binary_fill_holes(image0_sc_ch0_0)
    image0_sc_ch0_0_label = np.where(image0_sc_ch0_0_label, 1, 0)

    image_plot1_0 = imageio.imread(f'{baseDir}/Data/{cellMkr}/Snap-{sampleID}.tiff_files/{cellMkr_img}_sc_chan1_0.tif')
    image0_sc_ch1_0 =  np.zeros_like(image_plot1_0[...,1]) 
    image0_sc_ch1_0[image_plot1_0[...,1]>0] = 1
    image0_sc_ch1_0 = image0_sc_ch1_0.astype(float)
    image0_sc_ch1_0 = binary_closing(image0_sc_ch1_0)
    image0_sc_ch1_0_label = binary_fill_holes(image0_sc_ch1_0)

    image_plot2_0 = imageio.imread(f'{baseDir}/Data/{cellMkr}/Snap-{sampleID}.tiff_files/{cellMkr_img}_sc_chan2_0.tif')
    image0_sc_ch2_0 =  np.zeros_like(image_plot2_0[...,1]) 
    image0_sc_ch2_0[image_plot2_0[...,1]>0] = 1
    image0_sc_ch2_0 = image0_sc_ch2_0.astype(float)
    image0_sc_ch2_0 = binary_closing(image0_sc_ch2_0)
    image0_sc_ch2_0_label = binary_fill_holes(image0_sc_ch2_0)
    image0_sc_nuc = image0_sc_ch1_0_label+image0_sc_ch2_0_label

    image_plot_nuc = image_plot2_0+image_plot1_0
    file_path = f'/Volumes/T9/{cellMkr}/seven/test_results/image{train_idx}.pkl'
    with open(file_path, 'rb') as file:
        output = pickle.load(file)


    thresholds = np.arange(0.0, 1.0, 0.05)

    results = []

    for threshold in thresholds: 
        # marker 
        image_gray = output['marker']
        image_blurred = filters.gaussian(image_gray, sigma=1)
        binary_image = image_blurred > threshold
        marker_IoU = calculate_iou(image0_sc_ch0_0_label, binary_image)

        # nuclei
        image_gray = output['nuc_in']
        image_blurred = filters.gaussian(image_gray, sigma=1)
        binary_in = image_blurred > threshold
        # plt.imshow(binary_in)
        image_gray = output['nuc_ou']
        image_blurred = filters.gaussian(image_gray, sigma=1)
        binary_out = image_blurred > threshold
        # plt.imshow(binary_out)
        binary_nuc = binary_in+binary_out
        # plt.imshow(binary_nuc)
        nuclei_IoU = calculate_iou(image0_sc_nuc, binary_nuc)
        
        results.append([threshold, marker_IoU, nuclei_IoU])

    df = pd.DataFrame(results, columns=['Threshold', 'Marker_IoU', 'Nuclei_IoU'])

    df['Mean_IoU'] = df[['Marker_IoU', 'Nuclei_IoU']].mean(axis=1)


    # Save the DataFrame to a CSV file
    output_csv_path = f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/seven/image{train_idx}_IoU.csv'  # Modify as needed

    df.to_csv(output_csv_path, index=False)
    
    print(f'Results for Sample ID {sampleID} saved to {output_csv_path}')



# %% Summarizing IoU

import pandas as pd
from functools import reduce
cellMkr = 'PECAM'
# List of file paths
file_paths = [
    f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/seven/image0_IoU.csv',
    f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/seven/image1_IoU.csv',
    f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/seven/image2_IoU.csv',
    f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/seven/image3_IoU.csv',
    f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/seven/image4_IoU.csv',
    f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/seven/image5_IoU.csv',
    f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/seven/image6_IoU.csv'
]

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through the file paths, load each file, and select necessary columns
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    # Select only the relevant columns and rename them for merging
    df = df[['Threshold', 'Marker_IoU', 'Nuclei_IoU', 'Mean_IoU']]
    df.rename(columns={
        'Marker_IoU': f'Marker_IoU_{i}',
        'Nuclei_IoU': f'Nuclei_IoU_{i}',
        'Mean_IoU': f'Mean_IoU_{i}'
    }, inplace=True)
    dataframes.append(df)

# Merge all DataFrames on the 'Threshold' column using reduce
result_df = reduce(lambda left, right: pd.merge(left, right, on='Threshold'), dataframes)

# Compute the mean of IoU columns across all images for each type of IoU and store only the averages
for column_type in ['Marker_IoU', 'Nuclei_IoU', 'Mean_IoU']:
    # Create a list of column names for each IoU type across all files
    columns = [f'{column_type}_{i}' for i in range(len(file_paths))]
    # Calculate the mean across these columns and create a new column for the mean
    result_df[f'Average_{column_type}'] = result_df[columns].mean(axis=1)
    # Drop the individual IoU columns as they are no longer needed
    result_df.drop(columns=columns, inplace=True)




# Print the DataFrame to check the output
cellMkr = 'iba1'
cellMkr_img = 'pecam'
print(result_df)
output_csv_path = f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/seven/image_all_IoU.csv'  # Modify as needed
result_df.to_csv(output_csv_path, index=False)

# read the thresholding backin. 
input_csv_path = f'/Users/lesliemeng/ImPartial2/Data/{cellMkr}/four/image_all_IoU.csv'

result_df = pd.read_csv(input_csv_path)
print(result_df.head())

# Find the index of the maximum value for each IoU type
idx_marker_max = result_df['Average_Marker_IoU'].idxmax()
result_df.loc[idx_marker_max, 'Threshold']
result_df.loc[idx_marker_max, 'Average_Marker_IoU']

idx_nuclei_max = result_df['Average_Nuclei_IoU'].idxmax()
result_df.loc[idx_nuclei_max, 'Threshold']
result_df.loc[idx_nuclei_max, 'Average_Nuclei_IoU']

idx_mean_max = result_df['Average_Mean_IoU'].idxmax()
result_df.loc[idx_mean_max, 'Threshold']
result_df.loc[idx_mean_max, 'Average_Mean_IoU']




