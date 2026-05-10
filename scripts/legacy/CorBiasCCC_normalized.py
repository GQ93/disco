#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:08:39 2024

@author: lesliemeng
This file is to create scatter plot of existing deconvoluted results versus 
the ImPartial outputs, and calculate spearman correlation, sum of absolute bias,
and Lin's concordance after impartial results being normazlied. 

ENS deconvolution results are already normalized. 

"""

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

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from functools import reduce

# %% Section 0: Load ENSEMBLE and Patrick results
file_path = '/Users/lesliemeng/ImPartial2/Data/EnsDeconv_ROS.csv'
EnsDeconv = pd.read_csv(file_path)
print(EnsDeconv.head())
print(EnsDeconv['Unnamed: 0'])
EnsDeconv.rename(columns={'Unnamed: 0': 'subjID'}, inplace=True)
# 'SubID' 'Astro' 'Micro' 'Endo' 'Neuron' 'Oligo' 

# unnormalized patrick: Patric_results 
# normalized patrick: Patric_results_norm
file_path = '/Users/lesliemeng/ImPartial2/Data/Patric_results_norm.csv'
PatrickDeconv = pd.read_csv(file_path)


# %% Section 1: Parameter specifications
# oneSplitFour, one, oneSplitFour_newScribble, one_newScribble, oneSplitFour_newScribbleOval
testFig = 'oneSplitFour'


# neurons (NeuN), astrocytes (GFAP), microglia (IBA1),
# oligodendrocytes (OLIG2), endothelial (PECAM).

# NeuN 0.36 , GFAP 0.18 , iba1 0.07, Olig2 0.19, PECAM 0.12
ImpartialCell = 'PECAM'

# EndoCellType: 'SubID' 'Astro' 'Micro' 'Endo' 'Neuron' 'Oligo' 
BenchCell = 'Endo'


# load impartial results
file_path = f'/Users/lesliemeng/ImPartial2/Data/normalizedProp_{testFig}.csv'
ImpartialDeconv = pd.read_csv(file_path)


# %% merge results.
EnsDeconv_sub = EnsDeconv[['subjID',BenchCell]]

Patrick_sub = PatrickDeconv[['subjID',BenchCell]]

ImpartialDeconv = ImpartialDeconv[['subjID', ImpartialCell]]

merge_1 = [EnsDeconv_sub, Patrick_sub, ImpartialDeconv]
merge_1 = reduce(lambda left, right: pd.merge(left, right, on='subjID'), merge_1)
merge_1.columns = ['subjID','Ensemble','Patrick','ImPartial']

spearmanr(merge_1['Ensemble'],merge_1['ImPartial'])
spearmanr(merge_1['Ensemble'],merge_1['Patrick'])
spearmanr(merge_1['ImPartial'],merge_1['Patrick'])

sum(abs(merge_1['Ensemble']-merge_1['ImPartial']))
sum(abs(merge_1['Ensemble']-merge_1['Patrick']))
sum(abs(merge_1['ImPartial']-merge_1['Patrick']))


sum((merge_1['Ensemble']-merge_1['ImPartial'])**2)
sum((merge_1['Ensemble']-merge_1['Patrick'])**2)
sum((merge_1['ImPartial']-merge_1['Patrick'])**2)

ccc(merge_1['Ensemble'],merge_1['ImPartial'])
ccc(merge_1['Ensemble'],merge_1['Patrick'])
ccc(merge_1['ImPartial'],merge_1['Patrick'])


# %% Section 2 plotting
plt.figure(figsize=(6, 6))

# Scatter plot for ImPartial over Ensemble
plt.scatter(merge_1['Ensemble'], merge_1['ImPartial'], color='blue', alpha=0.4, edgecolors='w', label='ImPartial vs Ensemble')
plt.scatter(merge_1['Patrick'], merge_1['ImPartial'], color='red', alpha=0.4, edgecolors='w', label='ImPartial vs Patrick')
plt.scatter(merge_1['Ensemble'], merge_1['Patrick'], color='green', alpha=0.4, edgecolors='w', label='Ensemble vs Patrick')

# Set axis limits
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(f'{ImpartialCell}_Ensemble/Patrick')
plt.ylabel('ImPartial')
plt.title(f'{ImpartialCell}_{testFig}_norm')
# Add a 45-degree line
lims = [0, 1]  # Updated limits for the 45 degree line
plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='45 Degree Line')
plt.legend()

correlation, p_value = spearmanr(merge_1['Ensemble'],merge_1['ImPartial'])
abs_bias = sum(abs(merge_1['Ensemble']-merge_1['ImPartial']))
sqr_bias = sum((merge_1['Ensemble']-merge_1['ImPartial'])**2)
LinCCC = ccc(merge_1['Ensemble'],merge_1['ImPartial'])

plt.text(x=1, 
         y=0.75, 
         s=f'S.Cor.: {correlation:.3f}({p_value:.3f}) ABS:{abs_bias:.3f} SQR:{sqr_bias:.3f} CCC:{LinCCC:.3f}', 
         verticalalignment='bottom', 
         horizontalalignment='right', 
         fontsize=12, color='blue', style='italic')

correlation, p_value = spearmanr(merge_1['ImPartial'], merge_1['Patrick'])
abs_bias = sum(abs(merge_1['ImPartial']-merge_1['Patrick']))
sqr_bias = sum((merge_1['ImPartial']-merge_1['Patrick'])**2)
LinCCC = ccc(merge_1['ImPartial'],merge_1['Patrick'])

plt.text(x=1, 
         y=0.7, 
         s=f'S.Cor.: {correlation:.3f}({p_value:.3f}) ABS:{abs_bias:.3f} SQR:{sqr_bias:.3f} CCC:{LinCCC:.3f}',
         verticalalignment='bottom', 
         horizontalalignment='right', 
         fontsize=12, color='red', style='italic')


correlation, p_value = spearmanr(merge_1['Ensemble'], merge_1['Patrick'])
abs_bias = sum(abs(merge_1['Ensemble']-merge_1['Patrick']))
sqr_bias = sum((merge_1['Ensemble']-merge_1['Patrick'])**2)
LinCCC = ccc(merge_1['Ensemble'],merge_1['Patrick'])

plt.text(x=1, 
         y=0.65, 
         s=f'S.Cor.: {correlation:.3f}({p_value:.3f}) ABS:{abs_bias:.3f} SQR:{sqr_bias:.3f} CCC:{LinCCC:.3f}',
         verticalalignment='bottom', 
         horizontalalignment='right', 
         fontsize=12, color='green', style='italic')


plt.grid(False)
# Save the plot to a file
# _norm: only impartial is normalized. 
# _norm2: only both impartial and patrick are normalized. 
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{ImpartialCell}/{testFig}/Corr_{ImpartialCell}_{testFig}_norm2.png", format='png', dpi=300)  # Specify path, format, and DPI

# Show the plot
plt.show()
