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
cellType = 'PECAM'

# oneSplitFour, one, oneSplitFour_newScribble, one_newScribble
testFig = 'seven'
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
    testFile_path = row['input_dir']+'test_results/'+row['input_file']
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
    binary_image = image_blurred > threshold_mkr
    #plt.imshow(binary_image)
    # Label the image and count the objects
    labeled_image = measure.label(binary_image)
    #plt.imshow(labeled_image)
    marker_n = labeled_image.max()
    
    
    #    task = '1'; ix_class = 0
    #plt.imshow(output['nuc_in'])
    image_gray = output['nuc_in']
    #plt.imshow(image_gray)
    # Apply a Gaussian blur to the image to remove some of the noise
    image_blurred = filters.gaussian(image_gray, sigma=1)
    # plt.imshow(image_blurred)
    binary_image = image_blurred > threshold_mkr
    # plt.imshow(binary_image)
    # Label the image and count the objects
    labeled_image = measure.label(binary_image)
    #plt.imshow(labeled_image)
    nuc0_n = labeled_image.max()
    
    
    #task = '1'; ix_class = 1
    #plt.imshow(output['nuc_ou'])
    image_gray = output['nuc_ou']
    #plt.imshow(image_gray)
    # Apply a Gaussian blur to the image to remove some of the noise
    image_blurred = filters.gaussian(image_gray, sigma=1)
    #plt.imshow(image_blurred)
    # Threshold the image to binary using Otsu's method
    #threshold = filters.threshold_otsu(image_blurred)
    binary_image = image_blurred > threshold_nuc
    #plt.imshow(binary_image)
    labeled_image = measure.label(binary_image)
    #plt.imshow(labeled_image)
    nuc1_n = labeled_image.max()

    #### Save results
    prop1 = round(marker_n/(nuc0_n+nuc1_n),4)
    prop2 = round(nuc0_n/(nuc0_n+nuc1_n),4)
    
    # Store the results
    results.append([row['input_file'].replace('.npz', ''), prop2])
    

results[0:10]
# Create DataFrame once after loop
testResults = pd.DataFrame(results, columns=['FileIndex', 'Test_prop'])
pd_files
print(pd_files.columns)
pd_files['input_file']


#combined_df = pd.merge(testResults, pd_files, left_on='FileIndex', right_index=True)
#final_df = combined_df[['FileIndex', 'input_file', 'Test_prop']]
#print(final_df)
#Using .copy() is a great solution when you're slicing from another DataFrame and 
#plan to modify the new DataFrame independently. 
#makes a copy to avoid any SettingWithCopyWarning in subsequent operations.
final_df = testResults[8:].copy() # remove training image.
final_df[0:10]
# Split 'input_file' into two parts at the underscore and expand
final_df[['subjID', 'sampID']] = final_df['FileIndex'].str.split('_', n=1, expand=True)
# Only keep digits in 'image_number'
final_df['subjID'] = final_df['subjID'].str.extract('(\d+)')
# Extract the numeric part before the '.npz' extension
final_df['sampID'] = final_df['sampID'].str.extract('(\d+)')

mean_prop_by_subj = final_df.groupby('subjID')['Test_prop'].mean()

final_df
mean_prop_by_subj

with pd.ExcelWriter(f'/{base_dir}/{cellType}/{testFig}/output_data_{cellType}_{testFig}_IoU_0.5.xlsx') as writer:
    # Write final_df to a sheet named 'Full Data'
    final_df.to_excel(writer, sheet_name='Full Data', index=False)
    
    # Since mean_prop_by_subj is a Series, convert it to DataFrame and then write to a sheet named 'Mean Properties'
    mean_prop_by_subj.to_frame('Test_prop').reset_index().to_excel(writer, sheet_name='Mean Properties', index=False)
    
    
# %% Section 3: Testing image
## plotting
import matplotlib.pyplot as plt
# Set up the plotting environment
plt.figure(figsize=(12, 6))
# Create a bar plot
for key, grp in final_df.groupby('subjID'):
    plt.bar(grp['sampID'], grp['Test_prop'], label=f'Subj {key}')
# Label formatting to make the x-axis labels more readable
plt.xticks(rotation=90)
# Adding labels and title
plt.xlabel('Sample ID')
plt.ylabel('Test Property')
plt.title('Test Property by Sample ID, Clustered by Subject ID')
# Set y-axis limits
plt.ylim(0, 1)
# Add a legend at the bottom of the figure with vertical text
plt.legend(title='Subject ID', loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=7, handletextpad=0.5, columnspacing=1, frameon=False)
# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust the rectangle [left, bottom, right, top] in normalized (0, 1) figure coordinates.

# Show the plot
#plt.show()
# Save the plot as a PDF file
plt.savefig(f'{data_dir}/{cellType}_prop_output_{testFig}_mainFig.pdf')



# %% Section 4: plot main figure
## plotting
import matplotlib.pyplot as plt
# Set up the plotting environment
plt.figure(figsize=(12, 6))
# Create a bar plot
for key, grp in final_df.groupby('subjID'):
    plt.bar(grp['sampID'], grp['Test_prop'], label=f'Subj {key}')
# Label formatting to make the x-axis labels more readable
plt.xticks(rotation=90)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.xticks([])
# Adding labels and title
plt.xlabel('Sample')
plt.ylabel('Proportion')
#plt.title('Test Property by Sample ID, Clustered by Subject ID')
# Set y-axis limits
plt.ylim(0, 1)
# Add a legend at the bottom of the figure with vertical text
#plt.legend(title='Subject ID', loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=7, handletextpad=0.5, columnspacing=1, frameon=False)
# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust the rectangle [left, bottom, right, top] in normalized (0, 1) figure coordinates.

# Show the plot
#plt.show()
# Save the plot as a PDF file
plt.savefig(f'{data_dir}/{cellType}_prop_output_{testFig}_mainFig.png')