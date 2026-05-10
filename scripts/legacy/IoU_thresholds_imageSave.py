#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is to get sample level proportion based on thresholds tuned by IoU
@author: lesliemeng
"""
# %% load package
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, pickle
import sys
from skimage import io
import os
from skimage import io, filters, measure, morphology, color
#import general 
sys.path.append("../../")


%load_ext autoreload
%autoreload 2
%matplotlib inline
!cd /Users/lesliemeng/ImPartial2


# %% Section 0: Parameter specifications
# NeuN 0.36 , GFAP 0.18 , iba1 0.07, Olig2 0.19, PECAM 0.12
cellType = 'iba1'

# oneSplitFour, one, oneSplitFour_newScribble, one_newScribble
testFig = 'four'
base_dir = 'Users/lesliemeng/ImPartial2/Data'
testFilePath = '/Volumes/T9/' # 'Users/lesliemeng/ImPartial2/Data' /Volumes/Data 2

# %% Section 1: Data Loading
data_dir = f'/{base_dir}/{cellType}/{testFig}/' ## This should be the file where the .npz data is
pd_files = pd.read_csv(data_dir+'files_results.csv',index_col=0)
pd_files['input_dir'] = f'{testFilePath}{cellType}/{testFig}/'
print(pd_files)
scribbles = '150'
files_scribbles = data_dir + 'files_2tasks1x2classes_3images_scribble_train_' + scribbles + '_results.csv'
pd_files_scribbles = pd.read_csv(files_scribbles)
print('Total images  train: ', len(pd_files_scribbles),'; test: ', len(pd_files)-len(pd_files_scribbles))
pd_files_scribbles['prefix']

    
# %% Section 2: Testing image
import pandas as pd

# Initialize a list to store the data
results = []
# threshold = 0.90
threshold_mkr = 0.85
threshold_nuc = 0.5

for index, row in pd_files.iterrows():
    #index = 1  # Index for the second row (Python uses 0-based indexing)
    #index = 1586
    #row = pd_files.iloc[index]
    
    print(index)
    testFile_path = row['input_dir']+'test_results_iba1/'+row['input_file']
    testFile_path = testFile_path.replace('.npz', '.pkl')
    
    with open(testFile_path, 'rb') as file:
        output = pickle.load(file)
    
    #task = '0'; ix_class = 0
    #plt.imshow(output['marker'])
    image_gray = output['marker']
    # plt.imshow(image_gray)
    # Apply a Gaussian blur to the image to remove some of the noise
    image_blurred = filters.gaussian(image_gray, sigma=1)
    # plt.imshow(image_blurred)
    marker = image_blurred > threshold_mkr
    #plt.imshow(marker)
    
    #    task = '1'; ix_class = 0
    #plt.imshow(output['nuc_in'])
    image_gray = output['nuc_in']
    #plt.imshow(image_gray)
    # Apply a Gaussian blur to the image to remove some of the noise
    image_blurred = filters.gaussian(image_gray, sigma=1)
    # plt.imshow(image_blurred)
    nuc_in = image_blurred > threshold_mkr
    # plt.imshow(nuc_in)
    
    #task = '1'; ix_class = 1
    #plt.imshow(output['nuc_ou'])
    image_gray = output['nuc_ou']
    #plt.imshow(image_gray)
    # Apply a Gaussian blur to the image to remove some of the noise
    image_blurred = filters.gaussian(image_gray, sigma=1)
    #plt.imshow(image_blurred)
    # Threshold the image to binary using Otsu's method
    #threshold = filters.threshold_otsu(image_blurred)
    nuc_ou = image_blurred > threshold_nuc
    #plt.imshow(nuc_ou)
    
    output_dict = { "marker": marker, "nuc_in": nuc_in, "nuc_ou": nuc_ou}
    # Specify a full path and file name
    #full_path = row['input_dir']+'/test_results/'+row['input_file']
    savePath = testFile_path.replace('test_results_iba1', 'test_results_binary')

    with open(savePath, 'wb') as file:
        pickle.dump(output_dict, file)


with open('/Volumes/T9/iba1/four/test_results_binary/image11453772_7317.pkl', 'rb') as file:
    testOut = pickle.load(file)
    
plt.imshow(testOut['marker'])
plt.imshow(testOut['nuc_in'])
plt.imshow(testOut['nuc_ou'])

