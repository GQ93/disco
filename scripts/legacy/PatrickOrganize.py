#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:10:02 2024

@author: lesliemeng

This file is to organize patrick's deconvolution results. 
"""
import pandas as pd
from functools import reduce

# %% Section 0: organize all cell types into 1 file 
patrick_path = "/Users/lesliemeng/Library/CloudStorage/Dropbox/Case Western Reserve/Harry/IHC/CortexCellDeconv/CellTypeDeconvAnalysis/Data/"
wk_dir = "/Users/lesliemeng/ImPartial2/Data"

# Patric: 'astro' 'microglia' 'endo' 'neuro' 'oligo'

astro = pd.read_csv(f'{patrick_path}IHC.astro.txt',delimiter='\t', header=None, quotechar='"')
astro = astro.transpose()    
astro.columns = ['subjID', 'Astro']
astro['subjID'] = astro['subjID'].astype('int64')

microglia = pd.read_csv(f'{patrick_path}IHC.microglia.txt',delimiter='\t', header=None, quotechar='"')
microglia = microglia.transpose()    
microglia.columns = ['subjID', 'Micro']
microglia['subjID'] = microglia['subjID'].astype('int64')

endo = pd.read_csv(f'{patrick_path}IHC.endo.txt',delimiter='\t', header=None, quotechar='"')
endo = endo.transpose()    
endo.columns = ['subjID', 'Endo']
endo['subjID'] = endo['subjID'].astype('int64')

neuro = pd.read_csv(f'{patrick_path}IHC.neuro.txt',delimiter='\t', header=None, quotechar='"')
neuro = neuro.transpose()    
neuro.columns = ['subjID', 'Neuron']
neuro['subjID'] = neuro['subjID'].astype('int64')

oligo = pd.read_csv(f'{patrick_path}IHC.oligo.txt',delimiter='\t', header=None, quotechar='"')
oligo = oligo.transpose()    
oligo.columns = ['subjID', 'Oligo']
oligo['subjID'] = oligo['subjID'].astype('int64')



patric_all = [astro, microglia, endo, neuro, oligo]
merged_mean_all = reduce(lambda left, right: pd.merge(left, right, on='subjID'), patric_all)

merged_mean_all.to_csv(f'{wk_dir}/Patric_results.csv', index=False)

# %% Section 1: normalize the results. 

file_path = '/Users/lesliemeng/ImPartial2/Data/Patric_results.csv'
PatrickDeconv = pd.read_csv(file_path)
PatrickDeconv_norm = PatrickDeconv.copy()
# Columns to normalize
columns_to_normalize = ['Astro', 'Micro', 'Endo', 'Neuron', 'Oligo']

# Normalize each row
PatrickDeconv_norm[columns_to_normalize] = PatrickDeconv[columns_to_normalize].div(PatrickDeconv[columns_to_normalize].sum(axis=1), axis=0)
PatrickDeconv_norm.iloc[:, 1:].sum(axis=1)

wk_dir = "/Users/lesliemeng/ImPartial2/Data"
PatrickDeconv_norm.to_csv(f'{wk_dir}/Patric_results_norm.csv', index=False)


