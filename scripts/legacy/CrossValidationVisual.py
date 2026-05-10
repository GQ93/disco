#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 00:46:51 2024

@author: lesliemeng
"""
import matplotlib.pyplot as plt
import numpy as np

# %% Section 0: Parameter specifications
# NeuN 0.36 , GFAP 0.18 , iba1 0.07, Olig2 0.19, PECAM 0.12
cellType = 'iba1'
True_prop = 0.07

# oneSplitFour, one, oneSplitFour_newScribble, one_newScribble
testFig = 'three'
base_dir = 'Users/lesliemeng/ImPartial2/Data'
testFilePath = '/Volumes/T9/' # 'Users/lesliemeng/ImPartial2/Data' /Volumes/Data 2

0.0796
0.0956
0.1108
0.1096



values = [0.0796, 0.0956, 0.1108, 0.1096]

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(4, 6))  # Adjust the size as needed

# Adding jitter to the x-coordinates
x_coords = 1 + 0.05 * np.random.randn(len(values))  # Adds small random noise

# Plot each value as a point on the y-axis
ax.scatter(x_coords, values, color='blue')

# Adding labels and title
ax.set_title('Cross Validation')
ax.set_xlabel('iba1')  # x-axis label
ax.set_ylabel('Proportion')

# Set the limits for the x-axis and y-axis for better visualization
ax.set_xlim(0.5, 1.5)
ax.set_ylim(0, 1)  # Adjusted to better fit the data

# Remove x-ticks
ax.set_xticks([])  # Removes the x-ticks

# Label each point with its value
for x, y in zip(x_coords, values):
    ax.annotate(f'{y:.4f}',  # Format the value to 4 decimal places
                (x, y), 
                textcoords="offset points",  # Positioning the text
                xytext=(10,-10 if np.random.rand() > 0.5 else 10),  # Randomly position labels above or below the point
                ha='left')  # Horizontal alignment can be left, right, or center

# Add a horizontal line at y=0.07
ax.axhline(y=0.07, color='red', linestyle='--', linewidth=1)

# Show the plot
plt.show()