#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 23:48:02 2024

@author: lesliemeng
This file is to draw the pattern of Lin CCC, correlation, as the number of training 
images increases, specifically for iba1. 


"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np


# %% Lin's ccc and data analyze
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

def process_and_analyze_data(testFig, EndoCellType, cellType, patric_prop):
    # EnsDeconv
    ens_deconv_path = '/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/Jiebiao_share/EnsDeconv_ROS.csv'
    EnsDeconv = pd.read_csv(ens_deconv_path)
    EnsDeconv.rename(columns={'Unnamed: 0': 'subjID'}, inplace=True)
    EnsDeconv_sub = EnsDeconv[['subjID',EndoCellType]]
    
    # ImPartial
    impartial_path = f"/Users/lesliemeng/ImPartial2/Data/{cellType}/{testFig}/output_data_{cellType}_{testFig}.xlsx"
    full_data_df = pd.read_excel(impartial_path, sheet_name='Full Data')
    mean_properties_df = pd.read_excel(impartial_path, sheet_name='Mean Properties')
    merged_df = pd.merge(EnsDeconv_sub, mean_properties_df, on='subjID')
    merged_df = merged_df.rename(columns={'Test_prop': 'Test_I'})
    
    #Patrick
    patrick_path = "/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/CortexCellDeconv/CellTypeDeconvAnalysis/Data/"
    patrick_results = pd.read_csv(f'{patrick_path}IHC.{patric_prop}.txt', delimiter='\t', header=None, quotechar='"')
    patrick_results = patrick_results.transpose()
    patrick_results.columns = ['subjID', 'Test_P']
    patrick_results['subjID'] = patrick_results['subjID'].astype('int64')
    
    # Merge DataFrames
    merged_df2 = pd.merge(merged_df, patrick_results, on='subjID')
    merged_df2.columns = ['subjID', 'Ensemble', 'ImPartial', 'Patrick']

    # Statistical Analysis
    corr1, pval1 = spearmanr(merged_df2['Ensemble'], merged_df2['ImPartial'])
    corr2, pval2 = spearmanr(merged_df2['ImPartial'], merged_df2['Patrick'])
    corr3, pval3 = spearmanr(merged_df2['Ensemble'], merged_df2['Patrick'])

    ccc1 = ccc(merged_df2['Ensemble'], merged_df2['ImPartial'])
    ccc2 = ccc(merged_df2['ImPartial'], merged_df2['Patrick'])
    ccc3 = ccc(merged_df2['Ensemble'], merged_df2['Patrick'])

    # Organize results into a DataFrame
    data = {
        'Comparison': ['Ensemble vs ImPartial', 'ImPartial vs Patrick', 'Ensemble vs Patrick'],
        'Correlation': [corr1, corr2, corr3],
        'CCC': [ccc1, ccc2, ccc3]
    }
    correlation_df = pd.DataFrame(data)

    # Display the DataFrame
    print(correlation_df)

    return correlation_df


# %% Section 0: Parameter specifications
# neurons (NeuN), astrocytes (GFAP), microglia (IBA1),
# oligodendrocytes (OLIG2), endothelial (PECAM).

# NeuN 0.36 , GFAP 0.18 , iba1 0.07, Olig2 0.19, PECAM 0.12
cellType = 'iba1'

# EndoCellType: 'SubID' 'Astro' 'Micro' 'Endo' 'Neuron' 'Oligo' 
EndoCellType = 'Micro'

# Patric: 'astro' 'microglia' 'endo' 'neuro' 'oligo'
patric_prop = 'microglia'


# %% Section 1
# oneSplitFour, one, oneSplitFour_newScribble, one_newScribble, oneSplitFour_newScribbleOval

testFig = 'one_8406'
correlation_df1 = process_and_analyze_data(testFig, EndoCellType, cellType, patric_prop)
# iba1 tunning process. 
# 'one'
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.045918 -0.006402
#1   ImPartial vs Patrick     0.018469 -0.084070
#2    Ensemble vs Patrick     0.351633  0.017674
# 'one_8405'
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial    -0.005000 -0.000946
#1   ImPartial vs Patrick    -0.091939 -0.069531
#2    Ensemble vs Patrick     0.351633  0.017674
# one_8406
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial    -0.062857 -0.005101
#1   ImPartial vs Patrick    -0.381020 -0.314302
#2    Ensemble vs Patrick     0.351633  0.017674
# one_8407
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.160102  0.011044
#1   ImPartial vs Patrick    -0.127143 -0.081315
#2    Ensemble vs Patrick     0.351633  0.017674

testFig = 'two'
correlation_df2 = process_and_analyze_data(testFig, EndoCellType, cellType, patric_prop)
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.082551 -0.002024
#1   ImPartial vs Patrick     0.078163 -0.035674
#2    Ensemble vs Patrick     0.351633  0.017674

testFig = 'three'
correlation_df3 = process_and_analyze_data(testFig, EndoCellType, cellType, patric_prop)
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.038163 -0.003478
#1   ImPartial vs Patrick     0.260918  0.060851
#2    Ensemble vs Patrick     0.351633  0.017674

testFig = 'four'
correlation_df4 = process_and_analyze_data(testFig, EndoCellType, cellType, patric_prop)
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.030612 -0.000218
#1   ImPartial vs Patrick     0.277857  0.035394
#2    Ensemble vs Patrick     0.351633  0.017674

# %% Section 1 Plotting, four trainings
correlation_df1
correlation_df2
correlation_df3
correlation_df4
# Corresponding sample sizes
sample_sizes = [1, 2, 3,4]

# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
ccc_values = [
    correlation_df1[correlation_df1['Comparison'] == 'ImPartial vs Patrick']['CCC'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'ImPartial vs Patrick']['CCC'].values[0],
    correlation_df3[correlation_df3['Comparison'] == 'ImPartial vs Patrick']['CCC'].values[0],
    correlation_df4[correlation_df4['Comparison'] == 'ImPartial vs Patrick']['CCC'].values[0]
]
# Creating the plot
plt.figure(figsize=(5, 4))
plt.plot(sample_sizes, ccc_values, marker='o', linestyle='-', color='b')
plt.title('CCC: ImPartial vs Patrick')
plt.xlabel('Training Scribbles')
plt.ylabel('CCC')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylim(-0.4, 0.1) 
plt.subplots_adjust(left=0.20)
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/CCC_trend_impartial_vs_patrick_{cellType}.pdf", format='pdf')
plt.show()


# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
ccc_values = [
    correlation_df1[correlation_df1['Comparison'] == 'Ensemble vs ImPartial']['CCC'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'Ensemble vs ImPartial']['CCC'].values[0],
    correlation_df3[correlation_df3['Comparison'] == 'Ensemble vs ImPartial']['CCC'].values[0],
    correlation_df4[correlation_df4['Comparison'] == 'Ensemble vs ImPartial']['CCC'].values[0]
]
# Creating the plot
plt.figure(figsize=(5, 4))
plt.plot(sample_sizes, ccc_values, marker='o', linestyle='-', color='b')
plt.title('CCC: Ensemble vs ImPartial')
plt.xlabel('Training Scribbles')
plt.ylabel('CCC')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylim(-0.01, 0) 
plt.subplots_adjust(left=0.20)
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/CCC_trend_impartial_vs_ensemble_{cellType}.pdf", format='pdf')
plt.show()


# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
correlation_values = [
    correlation_df1[correlation_df1['Comparison'] == 'ImPartial vs Patrick']['Correlation'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'ImPartial vs Patrick']['Correlation'].values[0],
    correlation_df3[correlation_df3['Comparison'] == 'ImPartial vs Patrick']['Correlation'].values[0],
    correlation_df4[correlation_df4['Comparison'] == 'ImPartial vs Patrick']['Correlation'].values[0]
]
# Creating the plot
plt.figure(figsize=(5, 4))
plt.plot(sample_sizes, correlation_values, marker='o', linestyle='-', color='b')
plt.title('Correlation: ImPartial vs Patrick')
plt.xlabel('Training Scribbles')
plt.ylabel('Correlation')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(-0.5, 0.3) 
plt.subplots_adjust(left=0.20)
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/Correlation_trend_impartial_vs_patrick_{cellType}.pdf", format='pdf')
plt.show()



# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
correlation_values = [
    correlation_df1[correlation_df1['Comparison'] == 'Ensemble vs ImPartial']['Correlation'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'Ensemble vs ImPartial']['Correlation'].values[0],
    correlation_df3[correlation_df3['Comparison'] == 'Ensemble vs ImPartial']['Correlation'].values[0],
    correlation_df4[correlation_df4['Comparison'] == 'Ensemble vs ImPartial']['Correlation'].values[0]
]
# Creating the plot
plt.figure(figsize=(5, 4))
plt.plot(sample_sizes, correlation_values, marker='o', linestyle='-', color='b')
plt.title('Correlation: Ensemble vs ImPartial')
plt.xlabel('Training Scribbles')
plt.ylabel('Correlation')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(-0.1, 0.1) 
plt.subplots_adjust(left=0.20)
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/Correlation_trend_impartial_vs_ensemble_{cellType}.pdf", format='pdf')
plt.show()


# %% Section 1 Plotting three trainings
# neurons (NeuN), astrocytes (GFAP), microglia (IBA1),
# oligodendrocytes (OLIG2), endothelial (PECAM).
# NeuN 0.36 , GFAP 0.18 , iba1 0.07, Olig2 0.19, PECAM 0.12
cellType = 'GFAP'
# EndoCellType: 'SubID' 'Astro' 'Micro' 'Endo' 'Neuron' 'Oligo' 
EndoCellType = 'Astro'
# Patric: 'astro' 'microglia' 'endo' 'neuro' 'oligo'
patric_prop = 'astro'
testFig = 'one_6390'
correlation_df1 = process_and_analyze_data(testFig, EndoCellType, cellType, patric_prop)
# 'one'
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.190102  0.135907
#1   ImPartial vs Patrick     0.467857  0.113028
#2    Ensemble vs Patrick     0.324082  0.075888
# 'one_4201'
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.208980  0.090021
#1   ImPartial vs Patrick     0.030204 -0.003748
#2    Ensemble vs Patrick     0.324082  0.075888
# 'one_6390'
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial    -0.170918 -0.028357
#1   ImPartial vs Patrick    -0.143980 -0.013039
#2    Ensemble vs Patrick     0.324082  0.075888


testFig = 'two_5803_6390'
correlation_df2 = process_and_analyze_data(testFig, EndoCellType, cellType, patric_prop)
# 'two_5803_6390'
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.182755  0.123859
#1   ImPartial vs Patrick     0.235102  0.029597
#2    Ensemble vs Patrick     0.324082  0.075888
# two_4201_5803
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.213673  0.153409
#1   ImPartial vs Patrick     0.177551 -0.004312
#2    Ensemble vs Patrick     0.324082  0.075888

testFig = 'three'
correlation_df3 = process_and_analyze_data(testFig, EndoCellType, cellType, patric_prop)
#              Comparison  Correlation       CCC
#0  Ensemble vs ImPartial     0.268878  0.145110
#1   ImPartial vs Patrick     0.293265  0.022882
#2    Ensemble vs Patrick     0.324082  0.075888

# Corresponding sample sizes
sample_sizes = [1, 2, 3]


# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
ccc_values = [
    correlation_df1[correlation_df1['Comparison'] == 'ImPartial vs Patrick']['CCC'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'ImPartial vs Patrick']['CCC'].values[0],
    correlation_df3[correlation_df3['Comparison'] == 'ImPartial vs Patrick']['CCC'].values[0]
]
# Creating the plot
plt.figure(figsize=(5, 4))
plt.plot(sample_sizes, ccc_values, marker='o', linestyle='-', color='b')
plt.title('CCC: ImPartial vs Patrick')
plt.xlabel('Training Scribbles')
plt.ylabel('CCC')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(-0.03, 0.04) 
plt.subplots_adjust(left=0.20)
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/CCC_trend_impartial_vs_patrick_{cellType}.pdf", format='pdf')
plt.show()

# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
ccc_values = [
    correlation_df1[correlation_df1['Comparison'] == 'Ensemble vs ImPartial']['CCC'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'Ensemble vs ImPartial']['CCC'].values[0],
    correlation_df3[correlation_df3['Comparison'] == 'Ensemble vs ImPartial']['CCC'].values[0]
]
# Creating the plot
plt.figure(figsize=(5, 4))
plt.plot(sample_sizes, ccc_values, marker='o', linestyle='-', color='b')
plt.title('CCC: Ensemble vs ImPartial')
plt.xlabel('Training Scribbles')
plt.ylabel('CCC')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(-0.05, 0.2) 
plt.subplots_adjust(left=0.20)
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/CCC_trend_impartial_vs_ensemble_{cellType}.pdf", format='pdf')
plt.show()

# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
correlation_values = [
    correlation_df1[correlation_df1['Comparison'] == 'ImPartial vs Patrick']['Correlation'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'ImPartial vs Patrick']['Correlation'].values[0],
    correlation_df3[correlation_df3['Comparison'] == 'ImPartial vs Patrick']['Correlation'].values[0]
]
# Creating the plot
plt.figure(figsize=(5, 4))
plt.plot(sample_sizes, correlation_values, marker='o', linestyle='-', color='b')
plt.title('Correlation: ImPartial vs Patrick')
plt.xlabel('Training Scribbles')
plt.ylabel('Correlation')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(-0.2, 0.4) 
plt.subplots_adjust(left=0.20)
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/Correlation_trend_impartial_vs_patrick_{cellType}.pdf", format='pdf')
plt.show()

# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
correlation_values = [
    correlation_df1[correlation_df1['Comparison'] == 'Ensemble vs ImPartial']['Correlation'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'Ensemble vs ImPartial']['Correlation'].values[0],
    correlation_df3[correlation_df3['Comparison'] == 'Ensemble vs ImPartial']['Correlation'].values[0]
]
# Creating the plot
plt.figure(figsize=(5, 4))
plt.plot(sample_sizes, correlation_values, marker='o', linestyle='-', color='b')
plt.title('Correlation: Ensemble vs ImPartial')
plt.xlabel('Training Scribbles')
plt.ylabel('Correlation')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(-0.2, 0.3) 
plt.subplots_adjust(left=0.20)
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/Correlation_trend_impartial_vs_ensemble_{cellType}.pdf", format='pdf')
plt.show()


# %% Section 1 Plotting two trainings
correlation_df1

correlation_df2
# Corresponding sample sizes
sample_sizes = [1, 2]


# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
ccc_values = [
    correlation_df1[correlation_df1['Comparison'] == 'ImPartial vs Patrick']['CCC'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'ImPartial vs Patrick']['CCC'].values[0]
]
# Creating the plot
plt.figure(figsize=(8, 4))
plt.plot(sample_sizes, ccc_values, marker='o', linestyle='-', color='b')
plt.title('Trend of CCC for ImPartial vs Patrick across Training Images')
plt.xlabel('Training Scribbles')
plt.ylabel('CCC')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(-0.1, 0.1) 
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/CCC_trend_impartial_vs_patrick_{cellType}.pdf", format='pdf')
plt.show()


# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
ccc_values = [
    correlation_df1[correlation_df1['Comparison'] == 'Ensemble vs ImPartial']['CCC'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'Ensemble vs ImPartial']['CCC'].values[0]
]
# Creating the plot
plt.figure(figsize=(8, 4))
plt.plot(sample_sizes, ccc_values, marker='o', linestyle='-', color='b')
plt.title('Trend of CCC for Ensemble vs ImPartial across Training Images')
plt.xlabel('Training Scribbles')
plt.ylabel('CCC')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(-0.008, 0) 
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/CCC_trend_impartial_vs_ensemble_{cellType}.pdf", format='pdf')
plt.show()


# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
correlation_values = [
    correlation_df1[correlation_df1['Comparison'] == 'ImPartial vs Patrick']['Correlation'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'ImPartial vs Patrick']['Correlation'].values[0]
]
# Creating the plot
plt.figure(figsize=(8, 4))
plt.plot(sample_sizes, correlation_values, marker='o', linestyle='-', color='b')
plt.title('Trend of Correlation for ImPartial vs Patrick across Training Images')
plt.xlabel('Training Scribbles')
plt.ylabel('Correlation')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(0, 0.3) 
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/Correlation_trend_impartial_vs_patrick_{cellType}.pdf", format='pdf')
plt.show()



# Assuming correlation_df1, correlation_df2, correlation_df3 are already defined
# Extracting the CCC values for 'ImPartial vs Patrick' comparison from each DataFrame
correlation_values = [
    correlation_df1[correlation_df1['Comparison'] == 'Ensemble vs ImPartial']['Correlation'].values[0],
    correlation_df2[correlation_df2['Comparison'] == 'Ensemble vs ImPartial']['Correlation'].values[0]
]
# Creating the plot
plt.figure(figsize=(8, 4))
plt.plot(sample_sizes, correlation_values, marker='o', linestyle='-', color='b')
plt.title('Trend of Correlation for Ensemble vs ImPartial across Training Images')
plt.xlabel('Training Scribbles')
plt.ylabel('Correlation')
plt.xticks(sample_sizes)  # Ensure x-axis ticks represent the sample sizes
plt.grid(False)
plt.ylim(0, 0.1) 
plt.savefig(f"/Users/lesliemeng/ImPartial2/Data/{cellType}/Correlation_trend_impartial_vs_ensemble_{cellType}.pdf", format='pdf')
plt.show()