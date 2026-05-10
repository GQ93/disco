#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 of 3 — Train an ImPartial model and run inference.

Builds an :class:`impartial.Impartial_classes.ImPartialConfig` for the
selected cell type (``cellType`` near the top), then loads it as an
:class:`ImPartialModel` and runs ``general.training.eval`` on each test
``.npz`` image produced by ``01_TestingImages.py``. For every test image
the model returns three probability maps which are pickled to disk:

* ``marker``  — cell-type marker positivity
* ``nuc_in``  — DAPI nuclei *inside* marker-positive regions
* ``nuc_ou``  — DAPI nuclei *outside* marker-positive regions

Default training/inference hyperparameters (matching the manuscript)::

    seed                = 42
    GPU_ID              = 0
    LEARNING_RATE       = 5e-4
    EPOCHS              = 400
    nsaves              = 2          # number of checkpoint ensembles
    npatches_epoch      = 4096
    optim_weight_decay  = 1e-3
    weight_objectives   = {seg_fore: 0.45, seg_back: 0.45, rec: 0.10}
    n_channels          = 2          # marker + DAPI
    classification_tasks:
        '0' (marker):  classes=1, ncomponents=[2,2],   rec_channels=[0]
        '1' (nuclei):  classes=2, ncomponents=[1,1,2], rec_channels=[1]

External dependency
-------------------
This script imports ``impartial``, ``dataprocessing``, and ``general``
modules from the **ImPartial2** codebase (the U-Net-based weakly supervised
segmentation framework of Martinez et al., 2021, on which DISCO is built).
Clone that codebase separately and ensure it is importable on
``sys.path`` before running.

Inputs
------
* Test ``.npz`` images written by ``01_TestingImages.py``.
* ``files_results.csv`` — per-image manifest with at minimum an
  ``input_dir`` column.
* ``files_2tasks1x2classes_3images_scribble_train_150_results.csv`` — manifest
  of the scribble-annotated training images (with ``gt_index_task0``
  / ``gt_index_task1`` columns).

Outputs
-------
* ``<basedir>/<cellType>_test/config.json`` — saved training configuration.
* ``<basedir>/<cellType>_test/history.json`` — per-epoch losses
  (loss / rec / seg_fore / seg_back, train + val).
* ``<basedir>/<cellType>_test/weights_best.pth`` and
  ``model_val_best_save{0,1}.pth`` — checkpoint ensemble.
* ``<savPath>/test_results/<image-name>.pkl`` — three-channel probability
  maps per test image.

How to run
----------
Spyder / Jupyter cell-mode file. Execute ``# %% ...`` cells sequentially.
Edit the hardcoded paths near the top (``savPath``, ``data_dir``,
``basedir``, ``ImPartial2`` clone location) to match your environment first.

Originally written 2024-06-17 by lesliemeng (Guanqun Meng).
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
cellType = 'PECAM'
# Need to specify the path where to store the testing results. 
savPath = f'/Volumes/T9/{cellType}/'

# %%Input Directory
from dataprocessing.dataloaders import ImageBlindSpotDataset
data_dir = f'/Users/lesliemeng/ImPartial2/Data/disco_data/{cellType}/' 
pd_files = pd.read_csv(data_dir+'files_results.csv', index_col=0)
files_scribbles = data_dir + 'files_2tasks1x2classes_3images_scribble_train_150_results.csv'
pd_files_scribbles = pd.read_csv(files_scribbles)
print('Total images  train: ', len(pd_files_scribbles),'; test: ', len(pd_files)-len(pd_files_scribbles))


# %% Model name and directory
basedir = f'/Users/lesliemeng/ImPartial2/Data/disco_data/{cellType}/'
model_name = f'{cellType}_test'
mkdir(basedir)
mkdir(basedir+model_name+'/')

# %% Config Parameters, should run once to update the config parameters at local
seed = 42
GPU_ID = 0 # 0 [0,1]
LEARNING_RATE = 5e-4
nsaves = 2 # number of checkpoint ensembles or cycles
EPOCHS= 400 # 400 # this is set large and is not supposed to be the reason of stopping training
npatches_epoch=4096
MCdrop = False

weight_objectives = {'seg_fore':0.45,'seg_back':0.45,'rec':0.1}

## Input channels and classification tasks for MIBI2CH
n_channels = 2
classification_tasks = {'0': {'classes': 1, 'ncomponents': [2, 2],'rec_channels': [0]},
                        '1': {'classes': 2, 'ncomponents': [1, 1, 2],'rec_channels': [1]}}
optim_weight_decay = 1e-3
for task in classification_tasks.keys():
    # get list of corresponding gt indexes
    ix_labels_list = pd_files_scribbles['gt_index_task' + task].values[0]
    if ',' in ix_labels_list[1:-1]:
        ix_labels_list = ix_labels_list[1:-1].split(',')
    else:
        ix_labels_list = [ix_labels_list[1:-1]]
    classification_tasks[task]['ix_gt_labels'] = ix_labels_list

npatch_image_sampler = np.maximum(int(npatches_epoch/len(pd_files_scribbles)),6)

from impartial.Impartial_classes import ImPartialConfig
config = ImPartialConfig(basedir=basedir,
                         model_name=model_name,
                         nsaves = nsaves,
                         EPOCHS=EPOCHS,
                         seed=seed,
                         GPU_ID=GPU_ID,
                         npatches_epoch=npatches_epoch,
                         n_channels = n_channels,
                         classification_tasks=classification_tasks,
                         npatch_image_sampler = npatch_image_sampler,
                         MCdrop = MCdrop, optim_weight_decay=optim_weight_decay,
                         LEARNING_RATE=LEARNING_RATE,weight_objectives=weight_objectives)
config.save_json()


# %% Model
from impartial.Impartial_classes import ImPartialModel
from general.utils import load_json
# Path to the JSON configuration file
config = load_json(f'/Users/lesliemeng/ImPartial2/Data/disco_data/{cellType}/{cellType}_test/config.json')
config['basedir'] = f'/Users/lesliemeng/ImPartial2/Data/disco_data/{cellType}/'
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
with open(f"/Users/lesliemeng/ImPartial2/Data/disco_data/{cellType}/{cellType}_test/history.json", "r") as json_file:
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
# pd_files[1600:1602]
# pd_files
for index, row in pd_files[1600:1602].iterrows():
    # index = 1  # Index for the second row (Python uses 0-based indexing)
    # index = 1586
    # row = pd_files.iloc[index]
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
    full_path = savPath+'test_results/'+row['input_file']

    parts = full_path.split('.')
    file_path = parts[0]+'.pkl'  #The first part is the filename without the extension
    with open(file_path, 'wb') as file:
        pickle.dump(output_dict, file)


# %% visualize training results
cellType = 'NeuN'
testFig = 'one'
from PIL import Image
testImagePath = "/Volumes/T9"
with open(f"{savPath}/image0.pkl", 'rb') as file:
    loaded_output_dict = pickle.load(file)
loaded_output_dict.keys()
plt.imshow(loaded_output_dict['marker'])
plt.imshow(loaded_output_dict['nuc_in'])
plt.imshow(loaded_output_dict['nuc_ou'])




