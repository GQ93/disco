#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 05:18:54 2024
This file is to tune threshold for the testing results, probability maps, from 
impartial model. It additionally save the cell type proportion results and 
visualize sample and subject specific proportion information
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


# %% Functions
def image_seg(output, task, ix_class, threshold):
    """
    Process the segmentation, apply Gaussian blur, threshold, and count the objects.
    
    Args:
    output: The output data containing image segmentations.
    task: Marker task
    ix_class: Nuclei task (in marker / out marker)
    threshold: The threshold value for binary segmentation.
    
    Returns:
    int: The count of labeled objects in the image.
    """
    image_gray = output[task]['class_segmentation'][0, ix_class, ...]
    # plt.imshow(image_gray)  # Uncomment to visualize the grayscale image
    image_blurred = filters.gaussian(image_gray, sigma=1)
    # plt.imshow(image_blurred)  # Uncomment to visualize the blurred image
    binary_image = image_blurred > threshold
    # plt.imshow(binary_image)  # Uncomment to visualize the binary image
    labeled_image = measure.label(binary_image)
    # plt.imshow(labeled_image)  # Uncomment to visualize the labeled image
    return labeled_image.max()

# %% Section 0: Parameter specifications
# NeuN 0.36 , GFAP 0.18 , iba1 0.07, Olig2 0.19, PECAM 0.12
cellType = 'NeuN'
True_prop = 0.36

# oneSplitFour, one, oneSplitFour_newScribble, one_newScribble
testFig = 'zero'
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

## loading training results
file_path = data_dir+f'{cellType}_test/output_list.pkl'
with open(file_path, 'rb') as file:
    output_list = pickle.load(file)
type(output_list)

    

# %% Section 2: Cross validataion

thresholds = np.arange(0.0, 1.0, 0.05)
#thresholds = np.arange(0.01, 1.0, 0.001)

# Leave-one-out cross-validation on the first three lists
#scbl_indices = [0, 1, 2, 3]# Use only the first three indices for training
#scbl_indices = [0]
scbl_indices = [0, 1]
results_cv = pd.DataFrame(columns=['Threshold', 'Test_prop'])


for test_idx in scbl_indices:
    results = pd.DataFrame(columns=['Threshold', 'Prop1', 'Prop2'])
    
    # Scribble figure not spliting
    if len(scbl_indices) == 1:
        output = output_list[test_idx]
        for threshold in thresholds:
            task = '0'; ix_class = 0
            #In Python, after you use a keyword argument in a function call 
            #(task='0', ix_class=0), every argument that follows must also be named explicitly.
            marker_n = image_seg(output, task, ix_class, threshold)
            
            task = '1'; ix_class = 0
            nuc0_n = image_seg(output, task, ix_class, threshold)

            # threshold = 0.9
            task = '1'; ix_class = 1
            nuc1_n = image_seg(output, task, ix_class, threshold)
            #nuc1_n

            prop1 = round(marker_n/(nuc0_n+nuc1_n),4)
            prop2 = round(nuc0_n/(nuc0_n+nuc1_n),4)
            
            temp_df = pd.DataFrame({'Threshold': [threshold], 'Prop1': [prop1], 'Prop2': [prop2]})
            results = pd.concat([results, temp_df], ignore_index=True)
        # Calculate mean and find the optimal threshold
        mean_results = results.groupby('Threshold').agg({'Prop1': 'mean', 'Prop2': 'mean'}).reset_index()
        #mean_results['Prop1'] = abs(mean_results['Prop1'] - True_prop)
        #mean_results['Prop2'] = abs(mean_results['Prop2'] - True_prop)
        aa0 = mean_results[['Threshold', 'Prop2']].rename(columns={'Prop2': 'Test_prop'})
        results_cv = pd.concat([results_cv, aa0], ignore_index=True)
        #threshold_test = mean_results.loc[mean_results['Prop2'].idxmin(), 'Threshold']
        
        #results_cv.loc[len(results_cv)] = [threshold_test, prop2]
        
    # Scribble figure split into four. 
    else:
        train_indices = [i for i in scbl_indices if i != test_idx]
        # Loop over training indices and thresholds to find the best threshold
        for train_idx in train_indices:
            output = output_list[train_idx]
            for threshold in thresholds:
                
                task = '0'; ix_class = 0
                #In Python, after you use a keyword argument in a function call 
                #(task='0', ix_class=0), every argument that follows must also be named explicitly.
                marker_n = image_seg(output, task, ix_class, threshold)
                
                task = '1'; ix_class = 0
                nuc0_n = image_seg(output, task, ix_class, threshold)

                # threshold = 0.9
                task = '1'; ix_class = 1
                nuc1_n = image_seg(output, task, ix_class, threshold)
                #nuc1_n

                prop1 = round(marker_n/(nuc0_n+nuc1_n),4)
                prop2 = round(nuc0_n/(nuc0_n+nuc1_n),4)
                
                temp_df = pd.DataFrame({'Threshold': [threshold], 'Prop1': [prop1], 'Prop2': [prop2]})
                results = pd.concat([results, temp_df], ignore_index=True)
                
                
        # Calculate mean and find the optimal threshold
        mean_results = results.groupby('Threshold').agg({'Prop1': 'mean', 'Prop2': 'mean'}).reset_index()
        mean_results['Prop1'] = abs(mean_results['Prop1'] - True_prop)
        mean_results['Prop2'] = abs(mean_results['Prop2'] - True_prop)
        threshold_test = mean_results.loc[mean_results['Prop2'].idxmin(), 'Threshold']
            
            
        output = output_list[test_idx]
        task = '0'; ix_class = 0
        #plt.imshow(output[task]['class_segmentation'][0,ix_class,...])
        marker_n = image_seg(output, task, ix_class, threshold_test)

        task = '1'; ix_class = 0
        #plt.imshow(output[task]['class_segmentation'][0,ix_class,...])
        nuc0_n = image_seg(output, task, ix_class, threshold_test)

        task = '1'; ix_class = 1
        #plt.imshow(output[task]['class_segmentation'][0,ix_class,...])
        nuc1_n =  image_seg(output, task, ix_class, threshold_test)

        prop1 = round(marker_n/(nuc0_n+nuc1_n),4)
        prop2 = round(nuc0_n/(nuc0_n+nuc1_n),4)
        #prop2
        
        results_cv.loc[len(results_cv)] = [threshold_test, prop2]
    


results_cv   
results_cv['Diff'] = abs(results_cv['Test_prop'] - True_prop)
results_cv.loc[results_cv['Diff'].idxmin(), 'Threshold']

results = results_cv.transpose()    
results.to_csv(data_dir+f'image_processing_results_{cellType}_{testFig}.csv', index=True)


# %% Section 2: Testing image
import pandas as pd

# Initialize a list to store the data
results = []
threshold = results_cv.loc[results_cv['Diff'].idxmin(), 'Threshold']
# threshold = 0.90
#threshold = 0.5
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
    binary_image = image_blurred > threshold
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
    binary_image = image_blurred > threshold
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
    binary_image = image_blurred > threshold
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
final_df = testResults[3:].copy()
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

with pd.ExcelWriter(f'/{base_dir}/{cellType}/{testFig}/output_data_{cellType}_{testFig}.xlsx') as writer:
    # Write final_df to a sheet named 'Full Data'
    final_df.to_excel(writer, sheet_name='Full Data', index=False)
    
    # Since mean_prop_by_subj is a Series, convert it to DataFrame and then write to a sheet named 'Mean Properties'
    mean_prop_by_subj.to_frame('Test_prop').reset_index().to_excel(writer, sheet_name='Mean Properties', index=False)
    
    
# %% Section 2: Testing image
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
# plt.show()
# Save the plot as a PDF file
plt.savefig(f'{data_dir}/{cellType}_prop_output_{testFig}_mainFig.pdf')



# %% Section 2: plot main figure
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
# plt.show()
# Save the plot as a PDF file
plt.savefig(f'{data_dir}/{cellType}_prop_output_{testFig}_mainFig.png')





