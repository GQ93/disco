#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:12:26 2024

@author: lesliemeng

This script is to concatenate cell types for each subject, normalize cell prop.
to 1, and calculate respective correlations. 
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt

# %% Section 0: Parameter specifications
# NeuN 0.36 , GFAP 0.18 , iba1 0.07, Olig2 0.19, PECAM 0.12
#cellType = 'PECAM'

# EndoCellType: 'SubID' 'Astro' 'Micro' 'Endo' 'Neuron' 'Oligo' 
#EndoCellType = 'Endo'

# Patric: 'astro' 'microglia' 'endo' 'neuro' 'oligo'
#patric_prop = 'endo'

# oneSplitFour, one, oneSplitFour_newScribble, one_newScribble, oneSplitFour_newScribbleOval
testFig = 'one'

# neurons (NeuN), astrocytes (GFAP), microglia (IBA1),
# oligodendrocytes (OLIG2), endothelial (PECAM).


# %% Section 1: concatenate impartial results for each subject
# Extract Impartial Results
wk_dir = "/Users/lesliemeng/ImPartial2/Data"
file_path = f"{wk_dir}/NeuN/{testFig}/output_data_NeuN_{testFig}.xlsx"
# Load the "Full Data" sheet
full_data_df = pd.read_excel(file_path, sheet_name='Full Data')
# Load the "Mean Properties" sheet
mean_NeuN = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_NeuN.rename(columns={'Test_prop': 'NeuN'}, inplace=True)

file_path = f"{wk_dir}/GFAP/{testFig}/output_data_GFAP_{testFig}.xlsx"
mean_GFAP = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_GFAP.rename(columns={'Test_prop': 'GFAP'}, inplace=True)

file_path = f"{wk_dir}/iba1/{testFig}/output_data_iba1_{testFig}.xlsx"
mean_iba1 = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_iba1.rename(columns={'Test_prop': 'iba1'}, inplace=True)

file_path = f"{wk_dir}/Olig2/{testFig}/output_data_Olig2_{testFig}.xlsx"
mean_Olig2 = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_Olig2.rename(columns={'Test_prop': 'Olig2'}, inplace=True)

file_path = f"{wk_dir}/PECAM/{testFig}/output_data_PECAM_{testFig}.xlsx"
mean_PECAM = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_PECAM.rename(columns={'Test_prop': 'PECAM'}, inplace=True)

mean_all = [mean_NeuN, mean_GFAP, mean_iba1, mean_Olig2, mean_PECAM]
merged_mean_all = reduce(lambda left, right: pd.merge(left, right, on='subjID'), mean_all)


columns_to_normalize = ['NeuN', 'GFAP', 'iba1', 'Olig2', 'PECAM']
# make a deep copy of the original data
merged_mean_all_norm =  merged_mean_all.copy()
# normalization
merged_mean_all_norm[columns_to_normalize] = merged_mean_all[columns_to_normalize].div(merged_mean_all[columns_to_normalize].sum(axis=1), axis=0)
# check if normalized to 1
merged_mean_all_norm.iloc[:, 1:].sum(axis=1)


merged_mean_all_norm.to_csv(f'{wk_dir}/normalizedProp_{testFig}.csv', index=False)


# Generate the plot
merged_mean_all_order = merged_mean_all.sort_values(by='NeuN', ascending=True)
ax = merged_mean_all_order.set_index('subjID').plot(kind='bar', width=0.8, stacked=True, figsize=(14, 8))
ax.set_title(f'Stacked Cell Type Proportions per Subject {testFig}')
ax.set_ylim(bottom=0, top=2)
ax.set_xlabel('Subject ID')
ax.set_ylabel('Proportion')
# Customize x-axis labels to prevent overlap or improve readability
plt.xticks(rotation=90)  # Rotates labels to prevent overlap
# Show the plot
plt.tight_layout()  # Adjusts subplots to give some padding
plt.show()


merged_mean_all_2 = merged_mean_all.copy()
merged_mean_all_2['row_sum'] = merged_mean_all_2.iloc[:, 1:].sum(axis=1) 
merged_mean_all_2 = merged_mean_all_2.sort_values(by='row_sum', ascending=True)
merged_mean_all_2 = merged_mean_all_2.iloc[:, :-1]

column_mapping = {
    'NeuN': 'Neurons',
    'GFAP': 'Astrocytes',
    'iba1': 'Microglia',
    'Olig2': 'Oligodendrocytes',
    'PECAM': 'Endothelial'
}

# Rename the columns in the DataFrame
merged_mean_all_2.rename(columns=column_mapping, inplace=True)

# merged_mean_all_2.rename(columns={'Test_prop': 'PECAM'}, inplace=True)
ax = merged_mean_all_2.set_index('subjID').plot(
    kind='bar', 
    width=0.8, 
    stacked=True, figsize=(7, 8)) # width by tall
plt.tick_params(axis='y', which='major', labelsize=15)
ax.set_title(f'Stacked Cell Type Proportions per Subject {testFig}')
ax.set_ylim(bottom=0, top=2)
ax.set_xlabel('Subject ID')
ax.set_ylabel('Proportion')
# Adjust the legend to the new names, position outside the plot to avoid blocking data
ax.legend(title='Cell Type', loc='upper left')
# Customize x-axis labels to prevent overlap or improve readability
plt.xticks(rotation=90)  # Rotates labels to prevent overlap
# Show the plot
plt.tight_layout()  # Adjusts subplots to give some padding
plt.savefig(f'{wk_dir}/stacked_bar_plot_{testFig}.pdf', format='pdf', bbox_inches='tight')  # Save as PDF with tight bounding box
plt.show()

# %% Section 1: concatenate impartial results for each sample
# sample id not matched. 
wk_dir = "/Users/lesliemeng/ImPartial2/Data"
file_path = f"{wk_dir}/NeuN/{testFig}/output_data_NeuN_{testFig}.xlsx"
# Load the "Full Data" sheet
full_NeuN = pd.read_excel(file_path, sheet_name='Full Data')
# Load the "Mean Properties" sheet
mean_NeuN = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_NeuN.rename(columns={'Test_prop': 'NeuN'}, inplace=True)

file_path = f"{wk_dir}/GFAP/{testFig}/output_data_GFAP_{testFig}.xlsx"
full_GFAP = pd.read_excel(file_path, sheet_name='Full Data')
mean_GFAP = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_GFAP.rename(columns={'Test_prop': 'GFAP'}, inplace=True)

file_path = f"{wk_dir}/iba1/{testFig}/output_data_iba1_{testFig}.xlsx"
mean_iba1 = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_iba1.rename(columns={'Test_prop': 'iba1'}, inplace=True)

file_path = f"{wk_dir}/Olig2/{testFig}/output_data_Olig2_{testFig}.xlsx"
mean_Olig2 = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_Olig2.rename(columns={'Test_prop': 'Olig2'}, inplace=True)

file_path = f"{wk_dir}/PECAM/{testFig}/output_data_PECAM_{testFig}.xlsx"
mean_PECAM = pd.read_excel(file_path, sheet_name='Mean Properties')
mean_PECAM.rename(columns={'Test_prop': 'PECAM'}, inplace=True)

mean_all = [mean_NeuN, mean_GFAP, mean_iba1, mean_Olig2, mean_PECAM]
merged_mean_all = reduce(lambda left, right: pd.merge(left, right, on='subjID'), mean_all)







