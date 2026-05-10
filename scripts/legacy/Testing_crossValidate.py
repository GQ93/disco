#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 22:17:30 2024
Perform testing on the remaining images. The output should be three channels. 
Marker probability map, dapi inside marker probability map, and dapi outside
marker probability maps. 
@author: lesliemeng
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, pickle
import sys
from skimage import io
import os
#import general 
sys.path.append("../../")


%load_ext autoreload
%autoreload 2
%matplotlib inline
!cd /Users/lesliemeng/ImPartial2
# %%Input Cell information
cellType = 'iba1'
testFig = 'three_8405_8406_8407'
# saving path
# /Volumes/Data 2/Olig2/oneSplitFour_newScribbleOval/test_results
savPath = f'/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/' 
#/Users/lesliemeng/ImPartial2/Data/iba1/three_8265_8405_8406/test_results
# savPath = f'/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/'
#/Volumes/T9/GFAP/one
# testFig = 'one'

# %%Input Directory
from dataprocessing.dataloaders import ImageBlindSpotDataset
## This should be the file where the .npz data is:: !!!
data_dir = f'/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/' 
## This should be the file where the .npz data is
pd_files = pd.read_csv(data_dir+'files_results_v2.csv',index_col=0)
#pd_files['input_dir'] = data_dir
# pd_files.to_csv(data_dir + 'files_results.csv')
pd.set_option('display.max_colwidth', None)
print(pd_files)

# scribbles = '150'
# files_scribbles = data_dir + 'files_2tasks1x2classes_3images_scribble_train_' + scribbles + '_results.csv'
files_scribbles = data_dir + 'files_2tasks1x2classes_3images_scribble_train_150_results.csv'
pd_files_scribbles = pd.read_csv(files_scribbles)

## !!! Uncoment the following if 'input_dir' in pd_files_scribbles does not match data_dir
#pd_files_scribbles['input_dir'] = data_dir
# pd_files_scribbles.to_csv(files_scribbles,index=None)
print()
print('Total images  train: ', len(pd_files_scribbles),'; test: ', len(pd_files)-len(pd_files_scribbles))
pd_files_scribbles


pd.set_option('display.max_colwidth', None)
print(pd_files_scribbles['input_dir'])
print(pd_files_scribbles['scribble_file'])


# %% Model name and directory
from general.utils import mkdir
#basedir = '/data/natalia/models/Impartial/'
#/home/gxm324/Documents/IHC/ImPartial2
basedir = f'/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/'
model_name = f'{cellType}_test'

mkdir(basedir)
mkdir(basedir+model_name+'/')


# %% Model
from impartial.Impartial_classes import ImPartialModel
from general.utils import load_json
# Path to the JSON configuration file
config = load_json(f'/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/{cellType}_test/config.json')
print(config) 
config['basedir'] = f'/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/'
config['EPOCHS']
config['LEARNING_RATE']
# config_file_path = '/Users/lesliemeng/ImPartial2/Data/NeuN/oneSplitFour/NeuN_test/config.json'
# Load the JSON data from the file
# with open(config_file_path, 'r') as file:
#     config = json.load(file)
import torch
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        # Automatically set the DEVICE to GPU if CUDA is available, else CPU
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Config(**config)
config


im_model = ImPartialModel(config)

## load dataloader
im_model.load_dataloaders(data_dir,pd_files_scribbles,pd_files)
pd_files_scribbles
pd.set_option('display.max_colwidth', None)
print(pd_files_scribbles['input_dir'])

# %% Visualize some patches¶
## Visualize for reference
from general.utils import to_np
import seaborn as sns
color_list = sns.color_palette("husl", 9) #to color foreground scribbles

for batch, data in enumerate(im_model.dataloader_train):
    x = to_np(data['input']).transpose((0, 2, 3, 1))
    target = to_np(data['target']).transpose((0, 2, 3, 1))
    mask = to_np(data['mask']).transpose((0, 2, 3, 1))
    scribble = to_np(data['scribble']).transpose((0, 2, 3, 1))
    for item in np.arange(x.shape[0]):
        ix = 0
        for task in config.classification_tasks:
            classification_task = config.classification_tasks[task]
            nclasses = classification_task['classes']
            print('Task ' + str(task))
            aux = np.zeros([x.shape[1],x.shape[2],3])
            aux[...,1] = x[item,:,:,np.minimum(ix,x.shape[-1]-1)]*0.5
            aux[...,2] = scribble[item,:,:,ix+nclasses]
            for iclass in range(nclasses):
                aux = aux*(1-scribble[item,:,:,ix+iclass:ix+iclass+1]) + scribble[item,:,:,ix+iclass:ix+iclass+1]*color_list[iclass]
            plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            plt.title('Input')
            im_x = np.zeros([x.shape[1],x.shape[2],3])
            im_x[...,2] = x[item,:,:,0]
            im_x[...,1] = x[item,:,:,1]
            plt.imshow(im_x)
            plt.subplot(1,3,2)
            plt.title('Blind spots')
            plt.imshow(mask[item,:,:,np.minimum(ix,x.shape[-1]-1)])
            plt.subplot(1,3,3)
            plt.title('Scribbles')
            plt.imshow(aux)
            plt.show()
            ix += nclasses + 1
        break
    print(batch)
    break

# %%  Load Training Results
# load in training results
import json
with open(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/{cellType}_test/history.json", "r") as json_file:
    history = json.load(json_file)

tag = 'loss'
plt.figure(figsize=(8,8))
ix = 1
for tag in ['loss','rec','seg_fore','seg_back']:
    plt.subplot(2,2,ix)
    ix += 1
    plt.title(tag)
    plt.plot(history[tag+'_mbatch_train'],label='train')
    plt.plot(history[tag+'_val'],label='validation')
plt.show()



# %% Evaluate and save each testing image individually, save individually as well
# Here we only keep three channels, cell marker channel, nuclei in marker channel
# and nuclei out marker channel. 
# ------------ Dataloader ----------#
from dataprocessing.dataloaders import ToTensor,ImageSegDataset
from torchvision import transforms
import torch
from general.training import eval
import pickle
#pd_files[1334:]
#pd_files
for index, row in pd_files.iterrows():
    #index = 1  # Index for the second row (Python uses 0-based indexing)
    #index = 1586
    #row = pd_files.iloc[index]
    
    print(index,row)
    transforms_list = []
    if im_model.config.normstd:
        transforms_list.append(Normalize(mean=0.5, std=0.5))
    transforms_list.append(ToTensor())
    transform_eval = transforms.Compose(transforms_list)
    
    
    batch_size = 1
    #file_dir = pd_files.iloc[index:index+1]['input_dir'].item()
    file_dir = row['input_dir']
    dataloader_eval = torch.utils.data.DataLoader(ImageSegDataset(pd_files.iloc[index:index+1], file_dir, transform=transform_eval),
                                                batch_size=batch_size, shuffle=False, num_workers=8) ## Batch size 1 !!!
    output_list, gt_list = eval(dataloader_eval, im_model.model, im_model.optimizer, im_model.config, epoch=0, saveout=False, 
                                default_ensembles=True, model_ensemble_load_files=[])
    #type(output_list)
    #len(output_list)
    #type(output_list[0]) # check out the visualize 1 image section
    #plt.imshow(output_list[0]['0']['class_segmentation'][0,0,]) # marker
    #plt.imshow(output_list[0]['1']['class_segmentation'][0,0,]) # nuc in marker
    #plt.imshow(output_list[0]['1']['class_segmentation'][0,1,]) # nuc out marker

    marker = output_list[0]['0']['class_segmentation'][0, 0, ]
    nuc_in = output_list[0]['1']['class_segmentation'][0, 0, ]
    nuc_ou = output_list[0]['1']['class_segmentation'][0, 1, ]
    output_dict = {
        "marker": marker,
        "nuc_in": nuc_in,
        "nuc_ou": nuc_ou
    }
    # Specify a full path and file name
    #full_path = row['input_dir']+'/test_results/'+row['input_file']
    full_path = savPath+'/test_results/'+row['input_file']

    parts = full_path.split('.')
    file_path = parts[0]+'.pkl'  #The first part is the filename without the extension
    with open(file_path, 'wb') as file:
        pickle.dump(output_dict, file)


# %% visualize training results
cellType = 'NeuN'
testFig = 'one'
from PIL import Image
testImagePath = "/Volumes/T9"
with open(f"{testImagePath}/{cellType}/{testFig}/test_results/image0.pkl", 'rb') as file:
    loaded_output_dict = pickle.load(file)
loaded_output_dict.keys()
plt.imshow(loaded_output_dict['marker'])
plt.imshow(loaded_output_dict['nuc_in'])
plt.imshow(loaded_output_dict['nuc_ou'])
loaded_output_dict['marker'].shape
with open(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/{cellType}_test/output_list.pkl", 'rb') as file:
    outList = pickle.load(file) 
type(outList)
len(outList)
type(outList[0]) # check out the visualize 1 image section
plt.imshow(outList[0]['0']['class_segmentation'][0,0,]) # marker
plt.imshow(outList[0]['1']['class_segmentation'][0,0,]) # nuc in marker
plt.imshow(outList[0]['1']['class_segmentation'][0,1,]) # nuc out marker


with open(f"{testImagePath}/{cellType}/{testFig}/test_results/image0.pkl", 'rb') as file:
    loaded_output_dict = pickle.load(file)
loaded_output_dict.keys()
plt.imshow(loaded_output_dict['marker'])
plt.imshow(loaded_output_dict['nuc_ou'])
plt.imshow(loaded_output_dict['nuc_in'])



######
# marker
# Display and save the image without margins or axes
fig, ax = plt.subplots()
ax.axis('off')  # Turn off the axis
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove any padding around the image
# Assuming the data slice to be visualized
image_data = outList[0]['0']['class_segmentation'][0, 0]  # Adjust indexing as needed
ax.imshow(image_data, cmap='viridis')  # Use the colormap
# Save the figure to a TIFF file without any margins or axes
tiff_path = f"/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/image1_marker.tiff"
fig.savefig(tiff_path, format='tiff', bbox_inches='tight', pad_inches=0)
plt.close(fig)  # Close the plot to free up memory

# nuclei in
# Display and save the image without margins or axes
fig, ax = plt.subplots()
ax.axis('off')  # Turn off the axis
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove any padding around the image
# Assuming the data slice to be visualized
image_data = outList[0]['1']['class_segmentation'][0,0,]  # Adjust indexing as needed
ax.imshow(image_data, cmap='viridis')  # Use the colormap
# Save the figure to a TIFF file without any margins or axes
tiff_path = f"/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/image1_nuc_in.tiff"
fig.savefig(tiff_path, format='tiff', bbox_inches='tight', pad_inches=0)
plt.close(fig)  # Close the plot to free up memory

# nuclei out
# Display and save the image without margins or axes
fig, ax = plt.subplots()
ax.axis('off')  # Turn off the axis
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove any padding around the image
# Assuming the data slice to be visualized
image_data = outList[0]['1']['class_segmentation'][0,1,]  # Adjust indexing as needed
ax.imshow(image_data, cmap='viridis')  # Use the colormap
# Save the figure to a TIFF file without any margins or axes
tiff_path = f"/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/image1_mnuc_out.tiff"
fig.savefig(tiff_path, format='tiff', bbox_inches='tight', pad_inches=0)
plt.close(fig)  # Close the plot to free up memory
######
