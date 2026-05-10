#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 23:48:02 2024

@author: lesliemeng
This file is to create scatter plot of existing deconvoluted results versus 
the ImPartial outputs, and calculate spearman correlation, sum of absolute bias,
and Lin's concordance. 

"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from scipy.stats import pearsonr

# %% Lin's ccc
def ccc(x, y):
    # Calculate means of x and y
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    
    # Calculate variance of x and y
    s1_squared = np.var(x, ddof=0)  # https://numpy.org/doc/stable/reference/generated/numpy.var.html#numpy.var
    s2_squared = np.var(y, ddof=0)
    
    # Calculate covariance between x and y
    s12 = np.cov(x, y, ddof=0)[0, 1]  # Extract the covariance from the covariance matrix
    
    # Correct formula for Lin's CCC
    numerator = 2 * s12
    denominator = s1_squared + s2_squared + (mu_x - mu_y)**2
    ccc_value = numerator / denominator
    
    return ccc_value

# np.mean((merged_df2['ImPartial']-np.mean(merged_df2['ImPartial']))*(merged_df2['Patrick']-np.mean(merged_df2['Patrick'])))
# np.cov(merged_df2['Patrick'], merged_df2['ImPartial'], ddof=0)[0, 1]


# %% Section 0: Parameter specifications

# oneSplitFour, one, oneSplitFour_newScribble, one_newScribble, oneSplitFour_newScribbleOval
testFig = 'zero'

# neurons (NeuN), astrocytes (GFAP), microglia (IBA1),
# oligodendrocytes (OLIG2), endothelial (PECAM).

# NeuN 0.36 , GFAP 0.18 , iba1 0.07, Olig2 0.19, PECAM 0.12
cellType = 'NeuN'

# EndoCellType: 'SubID' 'Astro' 'Micro' 'Endo' 'Neuron' 'Oligo' 
EndoCellType = 'Neuron'

# Patric: 'astro' 'microglia' 'endo' 'neuro' 'oligo'
patric_prop = 'neuro'

# %% Section 1
file_path = '/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/Jiebiao_share/EnsDeconv_ROS.csv'
EnsDeconv = pd.read_csv(file_path)
print(EnsDeconv.head())
print(EnsDeconv['projid'])
EnsDeconv.rename(columns={'projid': 'subjID'}, inplace=True)
# 'SubID' 'Astro' 'Micro' 'Endo' 'Neuron' 'Oligo' 

EnsDeconv_sub = EnsDeconv[['subjID',EndoCellType]]



# Extract Impartial Results
file_path = f"/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/output_data_{cellType}_{testFig}.xlsx"
# Load the "Full Data" sheet
full_data_df = pd.read_excel(file_path, sheet_name='Full Data')
# Load the "Mean Properties" sheet
mean_properties_df = pd.read_excel(file_path, sheet_name='Mean Properties')
merged_df = pd.merge(EnsDeconv_sub, mean_properties_df, on='subjID')
merged_df = merged_df.rename(columns={'Test_prop': 'Test_I'})
type(merged_df)
print(merged_df.info())

# Patrick
patrick_path = "/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/CortexCellDeconv/CellTypeDeconvAnalysis/Data/"
patrick_results = pd.read_csv(f'{patrick_path}IHC.{patric_prop}.txt',delimiter='\t', header=None, quotechar='"')
patrick_results = patrick_results.transpose()    
patrick_results.columns = ['subjID', 'Test_P']
patrick_results['subjID'] = patrick_results['subjID'].astype('int64')



merged_df2 = pd.merge(merged_df, patrick_results, on='subjID')

merged_df2.columns = ['subjID','Ensemble','ImPartial','Patrick']

spearmanr(merged_df2['Ensemble'],merged_df2['ImPartial'])
spearmanr(merged_df2['ImPartial'],merged_df2['Patrick'])
spearmanr(merged_df2['Ensemble'],merged_df2['Patrick'])

pearsonr(merged_df2['Ensemble'], merged_df2['ImPartial'])
pearsonr(merged_df2['ImPartial'],merged_df2['Patrick'])
pearsonr(merged_df2['Ensemble'],merged_df2['Patrick'])


sum(abs(merged_df2['Ensemble']-merged_df2['ImPartial']))
sum(abs(merged_df2['ImPartial']-merged_df2['Patrick']))
sum(abs(merged_df2['Ensemble']-merged_df2['Patrick']))

sum((merged_df2['Ensemble']-merged_df2['ImPartial'])**2)
sum((merged_df2['ImPartial']-merged_df2['Patrick'])**2)
sum((merged_df2['Ensemble']-merged_df2['Patrick'])**2)

ccc(merged_df2['Ensemble'],merged_df2['ImPartial'])
ccc(merged_df2['ImPartial'],merged_df2['Patrick'])
ccc(merged_df2['Ensemble'],merged_df2['Patrick'])



# %% Section 2 plotting
plt.figure(figsize=(6, 6))

# Scatter plot for ImPartial over Ensemble
plt.scatter(merged_df2['Ensemble'], merged_df2['ImPartial'], color='blue', alpha=0.4, edgecolors='w', label='ImPartial vs Ensemble')
plt.scatter(merged_df2['Patrick'], merged_df2['ImPartial'], color='red', alpha=0.4, edgecolors='w', label='ImPartial vs Patrick')
plt.scatter(merged_df2['Ensemble'], merged_df2['Patrick'], color='green', alpha=0.4, edgecolors='w', label='Ensemble vs Patrick')

# Set axis limits
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(f'{EndoCellType}_Ensemble/Patrick')
plt.ylabel('ImPartial')
plt.title(f'{cellType}_{testFig}')
# Add a 45-degree line
lims = [0, 1]  # Updated limits for the 45 degree line
plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='45 Degree Line')
plt.legend()

correlation, p_value = spearmanr(merged_df2['Ensemble'],merged_df2['ImPartial'])
abs_bias = sum(abs(merged_df2['Ensemble']-merged_df2['ImPartial']))
sqr_bias = sum((merged_df2['Ensemble']-merged_df2['ImPartial'])**2)
LinCCC = ccc(merged_df2['Ensemble'],merged_df2['ImPartial'])

plt.text(x=1, 
         y=0.75, 
         s=f'S.Cor.: {correlation:.3f}({p_value:.3f}) ABS:{abs_bias:.3f} SQR:{sqr_bias:.3f} CCC:{LinCCC:.3f}', 
         verticalalignment='bottom', 
         horizontalalignment='right', 
         fontsize=12, color='blue', style='italic')

correlation, p_value = spearmanr(merged_df2['ImPartial'], merged_df2['Patrick'])
abs_bias = sum(abs(merged_df2['ImPartial']-merged_df2['Patrick']))
sqr_bias = sum((merged_df2['ImPartial']-merged_df2['Patrick'])**2)
LinCCC = ccc(merged_df2['ImPartial'],merged_df2['Patrick'])

plt.text(x=1, 
         y=0.7, 
         s=f'S.Cor.: {correlation:.3f}({p_value:.3f}) ABS:{abs_bias:.3f} SQR:{sqr_bias:.3f} CCC:{LinCCC:.3f}',
         verticalalignment='bottom', 
         horizontalalignment='right', 
         fontsize=12, color='red', style='italic')


correlation, p_value = spearmanr(merged_df2['Ensemble'], merged_df2['Patrick'])
abs_bias = sum(abs(merged_df2['Ensemble']-merged_df2['Patrick']))
sqr_bias = sum((merged_df2['Ensemble']-merged_df2['Patrick'])**2)
LinCCC = ccc(merged_df2['Ensemble'],merged_df2['Patrick'])

plt.text(x=1, 
         y=0.65, 
         s=f'S.Cor.: {correlation:.3f}({p_value:.3f}) ABS:{abs_bias:.3f} SQR:{sqr_bias:.3f} CCC:{LinCCC:.3f}',
         verticalalignment='bottom', 
         horizontalalignment='right', 
         fontsize=12, color='green', style='italic')


plt.grid(False)
# Save the plot to a file
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/Corr_{cellType}_{testFig}.png", format='png', dpi=300)  # Specify path, format, and DPI

# Show the plot
plt.show()

# %% plot for main figure
plt.figure(figsize=(6, 6))

# Scatter plot for ImPartial over Ensemble
plt.scatter(merged_df2['Ensemble'], merged_df2['ImPartial'], color='blue', alpha=0.4, edgecolors='w', label='ImPartial vs Ensemble', s=80)
plt.scatter(merged_df2['Patrick'], merged_df2['ImPartial'], color='red', alpha=0.4, edgecolors='w', label='ImPartial vs Patrick', s=80)
plt.scatter(merged_df2['Ensemble'], merged_df2['Patrick'], color='green', alpha=0.4, edgecolors='w', label='Ensemble vs Patrick', s=80)

# Set axis limits
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(f'{EndoCellType}_Ensemble/Patrick', fontsize=17)
plt.ylabel('ImPartial', fontsize=17)
#plt.title(f'{cellType}_{testFig}', fontsize=17)
# Add a 45-degree line
lims = [0, 1]  # Updated limits for the 45 degree line
plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='45 Degree Line')
#plt.legend(fontsize=15)

correlation, p_value = spearmanr(merged_df2['Ensemble'],merged_df2['ImPartial'])
abs_bias = sum(abs(merged_df2['Ensemble']-merged_df2['ImPartial']))
sqr_bias = sum((merged_df2['Ensemble']-merged_df2['ImPartial'])**2)
LinCCC = ccc(merged_df2['Ensemble'],merged_df2['ImPartial'])

# To change the size of the tick labels on both axes
plt.tick_params(axis='both', which='major', labelsize=20)

plt.text(x=0.9, 
         y=0.9, 
         s=f'S.Cor.: {correlation:.3f} ABD:{abs_bias:.3f} CCC:{LinCCC:.3f}', 
         verticalalignment='bottom', 
         horizontalalignment='right', 
         fontsize=16, color='blue', style='italic')

correlation, p_value = spearmanr(merged_df2['ImPartial'], merged_df2['Patrick'])
abs_bias = sum(abs(merged_df2['ImPartial']-merged_df2['Patrick']))
sqr_bias = sum((merged_df2['ImPartial']-merged_df2['Patrick'])**2)
LinCCC = ccc(merged_df2['ImPartial'],merged_df2['Patrick'])

plt.text(x=0.9, 
         y=0.85, 
         s=f'S.Cor.: {correlation:.3f} ABD:{abs_bias:.3f} CCC:{LinCCC:.3f}',
         verticalalignment='bottom', 
         horizontalalignment='right', 
         fontsize=16, color='red', style='italic')


correlation, p_value = spearmanr(merged_df2['Ensemble'], merged_df2['Patrick'])
abs_bias = sum(abs(merged_df2['Ensemble']-merged_df2['Patrick']))
sqr_bias = sum((merged_df2['Ensemble']-merged_df2['Patrick'])**2)
LinCCC = ccc(merged_df2['Ensemble'],merged_df2['Patrick'])

plt.text(x=0.9, 
         y=0.8, 
         s=f'S.Cor.: {correlation:.3f} ABD:{abs_bias:.3f} CCC:{LinCCC:.3f}',
         verticalalignment='bottom', 
         horizontalalignment='right', 
         fontsize=16, color='green', style='italic')


plt.grid(False)
# Save the plot to a file
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/Corr_{cellType}_{testFig}_mainplot.png", format='png', dpi=300)  # Specify path, format, and DPI

# Show the plot
plt.show()





