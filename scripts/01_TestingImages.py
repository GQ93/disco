"""
Step 1 of 3 — Data preparation: assemble two-channel ``.npz`` test images.

For each ROSMAP subject and sample, this script reads the DAPI (nuclei) and
cell-type-marker fluorescence channels stored as separate TIFF files, applies
``csbdeep.utils.normalize`` with ``p1=1, p99=99.8, clip=True`` to each channel,
stacks them along the last axis, and writes the result as a 2-channel
``.npz`` file consumable by step 2 (``02_Testing.py``).

The cell type to process is selected by the ``cell_marker`` variable near the
top of the file (one of ``NeuN``, ``GFAP``, ``iba1``, ``Olig2``, ``PECAM``).
The companion ``org_chan`` list specifies the channel filename suffixes used
to match TIFF files for that marker.

Inputs
------
* Per-subject TIFF directories under ``base_path/<SubID>/<Grey|Gray>/``
  with file pattern ``Snap-<sampleID>.tiff_files/Snap-<sampleID><suffix>*.tiff``.
* ``EnsDeconv_ROS.csv`` — used only to obtain the list of subject IDs to
  iterate over (column ``Unnamed: 0`` is renamed to ``SubID``; the first 49
  IDs are processed).

Outputs
-------
* ``output_dir/image<SubID>_<sampleID>.npz`` for each (subject, sample) pair,
  containing keys ``image`` (H, W, 2) and ``label`` (H, W, 3).
* ``output_dir/testFiles.csv`` listing all generated ``.npz`` filenames.

How to run
----------
This is a Spyder / Jupyter cell-mode file: execute its ``# %% ...`` cells
sequentially in an IPython environment. It will not run end-to-end as
``python 01_TestingImages.py`` because the source paths
(``/Users/lesliemeng/...``, the EnsDeconv CSV location) are hardcoded for
the original development machine — edit them to match your environment
before running.

Originally written 2024-05-28 by lesliemeng (Guanqun Meng).
"""
# %% Define variables
# NeuN 0.36 ['_dapi','_NeuN'], 
# GFAP 0.18 , 
# iba1 0.07 ['_b0c0x0','_b0c1x0'], 
# Olig2 0.19
# PECAM 0.12
cell_marker = "PECAM"

org_chan = ['_b0c0x0','_b0c1x0']

# %% Grab Ensenble infor. 
import pandas as pd
file_path = '/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/Jiebiao_share/EnsDeconv_ROS.csv'
EnsDeconv = pd.read_csv(file_path)
print(EnsDeconv.head())
print(EnsDeconv['Unnamed: 0'])
EnsDeconv.rename(columns={'Unnamed: 0': 'SubID'}, inplace=True)
topSubID = EnsDeconv['SubID'].head(49) #topSubID
# Print the result
print(topSubID)
print(EnsDeconv['SubID'])

# %% Combine original 2 channels.
import imageio
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob
base_path = f'/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/Jiebiao_share/{cell_marker}'
output_dir = f'/Users/lesliemeng/ImPartial2/Data/{cell_marker}/test_image'

#results_cv = pd.DataFrame(columns=['Threshold', 'Test_prop'])
topSubID[0:3]
# loop through subject IDs
for subject in topSubID:
    print(subject)
    
    if os.path.exists(f'{base_path}/{subject}/Grey/'):
        GreyOrGray = 'Grey'
    else:
        GreyOrGray = 'Gray'

    # read all file name under a subject
    # and get sample ID
    fileInSubID = os.listdir(f'{base_path}/{subject}/{GreyOrGray}/')
    pattern = re.compile(r'Snap-(\d+)\.tiff_files')
    # Create a set to store distinct numbers
    sampleID = set()
    # Iterate through files and apply regex to extract numbers
    for file in fileInSubID:
        match = pattern.search(file)
        if match:
            sampleID.add(int(match.group(1)))  # Convert to int to avoid leading zeros
    sampleID = sorted(sampleID)
    print(sampleID)
    
    # loop through samples, manipulate sample images, combine channels, and save
    for fileID in sampleID:
        # nuclei image in
        # NeuN
        #filePattern = f'{base_path}/{subject}/{GreyOrGray}/Snap-{fileID}.tiff_files/Snap-{fileID}_dapi*.tiff'
        filePattern = f'{base_path}/{subject}/{GreyOrGray}/Snap-{fileID}.tiff_files/Snap-{fileID}{org_chan[0]}*.tiff'
        filePattern = glob.glob(filePattern)
        Dapi_test = imageio.v3.imread(filePattern[0])
        # plt.imshow(Dapi_test[...,0])
        # plt.show()
        Dapi_test = normalize(Dapi_test[...,0],pmin=1,pmax=99.8,clip = True)
        #plt.imshow(Dapi_test)
        #plt.show()
        # cell image in
        #filePattern = f'{base_path}/{subject}/{GreyOrGray}/Snap-{fileID}.tiff_files/Snap-{fileID}_NeuN*.tiff'
        filePattern = f'{base_path}/{subject}/{GreyOrGray}/Snap-{fileID}.tiff_files/Snap-{fileID}{org_chan[1]}*.tiff'
        filePattern = glob.glob(filePattern)
        Cell_test = imageio.imread(filePattern[0])
        #plt.imshow(Cell_test[...,0])
        #plt.show()
        Cell_test = normalize(Cell_test[...,0],pmin=1,pmax=99.8,clip = True)
        #plt.imshow(Cell_test)
        #plt.show()
        
        image = np.concatenate([Cell_test[...,np.newaxis], Dapi_test[...,np.newaxis]],axis = -1)
        label = np.zeros(Dapi_test.shape + (3,))  # Add a tuple for the channel dimension
        np.savez(f"{output_dir}/image{subject}_{fileID}.npz",image = image,label = label)




# %% grab file name
import pandas as pd
testFiles = os.listdir(output_dir)
npz_files = [file for file in testFiles if file.endswith('.npz')]

# Sort the filtered list of filenames
npz_files = sorted(npz_files)

# Create a DataFrame from the sorted list
df = pd.DataFrame(npz_files, columns=['testFiles'])
df.to_csv(output_dir + '/testFiles.csv')



