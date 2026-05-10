#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:59:18 2024

@author: lesliemeng
"""
# %% load package
import matplotlib.pyplot as plt
import pandas as pd
import os, pickle
from skimage import io


# %% Section 0: Parameter specifications
# NeuN 0.36 , GFAP 0.18 , iba1 0.07, Olig2 0.19, PECAM 0.12
cellType = 'PECAM'

# oneSplitFour, one, oneSplitFour_newScribble, one_newScribble
testFig = 'one'
base_dir = 'Users/lesliemeng/ImPartial2/Data'
data_dir = f'/{base_dir}/{cellType}/{testFig}/' 

final_df = pd.read_excel(f'/{base_dir}/{cellType}/{testFig}/output_data_{cellType}_{testFig}.xlsx', sheet_name='Full Data')  

# %% Section 2: plot main figure
## plotting
# Set up the plotting environment
plt.figure(figsize=(12, 6))
# Create a bar plot
final_df['sampID'] = final_df['sampID'].astype(str)
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
plt.savefig(f'{data_dir}{cellType}_prop_output_{testFig}_mainFig.png')


import seaborn as sns
import matplotlib.pyplot as plt
# Use seaborn to plot a clustered bar chart
sns.barplot(data=final_df, x='sampID', y='Test_prop', hue='subjID')

# Improve the legibility of the plot
plt.xticks(rotation=90)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.ylim(0, 1)