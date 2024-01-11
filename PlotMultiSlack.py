# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:53:01 2023

@author: soere
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:57:13 2023

@author: soere
"""
import PyMGA
from PyMGA.utilities.plot import near_optimal_space_2D, near_optimal_space_matrix,set_options
import numpy as np
import yaml
import pandas as pd
import statistics 
import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
import dill
import os
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D


def interpolate_color(start_color, end_color, steps):
    start_rgb = np.array([int(start_color[i:i+2], 16) for i in (0, 2, 4)])
    end_rgb = np.array([int(end_color[i:i+2], 16) for i in (0, 2, 4)])

    colors = [list(map(int, start_rgb + (end_rgb - start_rgb) * i / steps)) for i in range(steps)]
    return ['#%02x%02x%02x' % tuple(c) for c in colors]


# Original data
marker_styles_lines = [''] * 10

# Define colors
start_color = 'FFB6C1'  # Light red
end_color = 'FF0000'  # Strong red
num_steps = len(marker_styles_lines)
interpolated_colors = interpolate_color(end_color, start_color, num_steps)

light_blue = 'ADD8E6'  # A light blue color
strong_blue = '0000FF'  # The strongest blue color
interpolated_colors_2 = interpolate_color(strong_blue, light_blue, num_steps)

light_green = '90EE90'  # A light green color
strong_green = '008000'  # The strongest green color
interpolated_colors_3 = interpolate_color(strong_green, light_green, num_steps)

light_purple = '9370DB'  # A light purple color
strong_purple = '800080'  # The strongest purple color
interpolated_colors_4 = interpolate_color(strong_purple, light_purple, num_steps)


Slacks = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
Years = [0.7, 0.3,0.1,0.05]


legend_labels = []
legend_handles = []


# Set MAA variables to explore
variables = {'Conventional': ['Generator', # Component type
                    ['Natural gas','Natural gas CCS','Nuclear','NuclearWTes'], # Component carriers
                    'p_nom',], # Component variable to explore
            'Wind+Solar': ['Generator',
                    ['Wind','Solar'],
                    'p_nom',],
                    } 

chosen_variables = ['Conventional', 'Wind+Solar']
all_variables    = list(variables.keys())

for i in range(len(Years)):
    
    if i == 0:
        Colors = interpolated_colors
    if i == 1:
        Colors = interpolated_colors_2
    if i == 2:
        Colors = interpolated_colors_3
    if i == 3:
        Colors = interpolated_colors_4
    
    for j in range(len(Slacks)):
    
        # Load verticies
        
        year = Years[i]
        slack = Slacks[j]
        file_name = f'Verticies'
        folder_name = f'NearOpt_Renew_Nuc_{slack}_{year}'
        file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\NearOpt_Renew_Nuc_MultiSlack\{folder_name}\{file_name}'
        
        verticies = np.loadtxt(file_path, delimiter=',', dtype=int)
        
        
        
        
        all_variables    = list(variables.keys())
        
        #plt.figure(47)
        
        # Set plotting options
        
        set_options()
        
        # Dataframe with all verticies
        verticies_df = pd.DataFrame(verticies,
                                    columns = all_variables)
        
        # Get verticies from only the chosen variables
        variable_verticies = verticies_df[chosen_variables]
        
        
        # Set x and y to be verticies for the first two variables
        x, y = variable_verticies[chosen_variables[0]], variable_verticies[chosen_variables[1]]
        
        
        # --------  Plot hull --------------------
        
        hull = ConvexHull(variable_verticies.values)
        
        # plot simplexes
        for simplex in hull.simplices:
            l0, = plt.plot(variable_verticies.values[simplex, 0], variable_verticies.values[simplex, 1], 'k-', 
                    color = Colors[j],marker=marker_styles_lines[j],
                    linewidth = 2, zorder = 0)
        
        
        plt.xlabel('Nuclear capacity [MW]')
        plt.ylabel('Wind + Solar capacity [MW]')
        plt.title('Near optimal spaces - Greenfield')
        plt.xlim(-1000, 80000)


Years = [0.7, 0.3,0.1,0.05]
#Years = [0.3,0.05]
marker_styles_dots = ['o']*len(Years)

Opt_Sol_Nuc = np.zeros(len(Years))
Opt_Sol_Ren = np.zeros(len(Years))
Brown_Opt_Sol_Nuc = np.zeros(len(Years))
Brown_Opt_Sol_Ren = np.zeros(len(Years))

   
for i in range(len(Years)):
    
    if i == 0:
        Colors = interpolated_colors
    if i == 1:
        Colors = interpolated_colors_2
    if i == 2:
        Colors = interpolated_colors_3
    if i == 3:
        Colors = interpolated_colors_4
    
    year = Years[i]
    folder_name = f'NearOpt_Renew_Nuc_{year}'
    file_name = f'OptimalSolution'
    file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
    
    Opt_Sol = np.loadtxt(file_path, delimiter=',', dtype=int)
    
    # For the line plot
    Opt_Sol_Nuc[i] = Opt_Sol[0]
    Opt_Sol_Ren[i] = Opt_Sol[1]
    
    file_name = f'BrownField_OptimalSolution'
    folder_name = f'BrownField_Nuc_{year}'
    file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\Myopic\{folder_name}\{file_name}'
   
    Brown_Opt_Sol = np.loadtxt(file_path, delimiter=',', dtype=int)
    
    Brown_Opt_Sol_Nuc[i] = Brown_Opt_Sol[0]
    Brown_Opt_Sol_Ren[i] = Brown_Opt_Sol[1]
    
    # Also plot the optimal point
    plt.scatter(Opt_Sol[0],Opt_Sol[1],color=Colors[0],s=200 , 
            marker=marker_styles_dots[i],edgecolor='green',
            linewidths=5)
    
    # Also plot the Brownfield results
    plt.scatter(Brown_Opt_Sol[0],Brown_Opt_Sol[1],color=Colors[0],s=200 , 
            marker=marker_styles_dots[i],edgecolor='brown',
            linewidths=5)
    



# plt.legend(handles=legend_handles, labels=legend_labels, 
#            loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=3)

# Manually specify the legend entries to display
legend_entries = [
    plt.Line2D([0], [0], color='red', markersize=10, linestyle='-'),
    plt.Line2D([0], [0], color='blue', markersize=10, linestyle='-'),
    plt.Line2D([0], [0], color='green', markersize=10, linestyle='-'),
    plt.Line2D([0], [0], color='purple', markersize=10, linestyle='-')
]

# Specify the labels for the legend entries
legend_labels = ['30% Reduction', '70% Reduction', '90% Reduction','95% Reduction']
#legend_labels = ['70% Reduction','95% Reduction']

# Display the legend with specific entries
plt.legend(handles=legend_entries, labels=legend_labels)

# plt.legend(handles=legend_handles, labels=legend_labels, 
#            loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=3)

# Also connect the dots
plt.plot(Opt_Sol_Nuc, Opt_Sol_Ren, color='black', marker=marker_styles_dots[i], linewidth=3)
plt.plot(Brown_Opt_Sol_Nuc, Brown_Opt_Sol_Ren, color='black', marker=marker_styles_dots[i], linewidth=3)


#plt.xlim(-1000,100000)
plt.title('Contours of near optimal solution space')
# plt.legend(['0% reduction','10% reduction','20% reduction','30% reduction',
#             '40% reduction','60% reduction','78% reduction',
#            '86% reduction','90% reduction'])




    
    
    
    