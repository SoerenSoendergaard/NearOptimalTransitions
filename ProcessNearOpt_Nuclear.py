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

def interpolate_color(start_color, middle_color, end_color, steps):
    start_rgb = np.array([int(start_color[i:i+2], 16) for i in (0, 2, 4)])
    middle_rgb = np.array([int(middle_color[i:i+2], 16) for i in (0, 2, 4)])
    end_rgb = np.array([int(end_color[i:i+2], 16) for i in (0, 2, 4)])

    colors = [
        list(map(int, start_rgb + (middle_rgb - start_rgb) * i / (steps // 2))) for i in range(steps // 2)
    ] + [
        list(map(int, middle_rgb + (end_rgb - middle_rgb) * i / (steps // 2))) for i in range(steps // 2 + 1)
    ]

    return ['#%02x%02x%02x' % tuple(c) for c in colors]

# Original data

Co2_Limits = [1,0.70,0.6,0.5,0.3,0.22,0.15,0.1,0.05, 0.02,0.01]
Co2_Limits_2 = [round(1 - x, 2) for x in Co2_Limits]
Years = [1,0.70,0.6,0.5,0.3,0.22,0.15,0.1,0.05, 0.02,0.01] # These are all the results ready for post processing
#Years = [1,0.3,0.05] #

marker_styles = [''] * len(Years)

# Define colors
start_color = '000000'  # black
middle_color = 'FF0000'  # red
end_color = 'F00FF0'   # green
num_steps = len(marker_styles) - 1
interpolated_colors = interpolate_color(start_color, middle_color, end_color, num_steps)


## Post process greenspaces

#Colors = ['black','black','red','red','blue','blue','green','green','purple']
Colors = interpolated_colors
marker_styles_dots = ['o','s','o','s','o','s','o','s','o']
marker_styles_dots = ['o']*len(Years)
marker_styles_lines = [''] * len(Years)
legend_labels = []
legend_handles = []
#Years = [2020,2035,2045]
#Co2_Limits = [1,0.4,0.1]
#Colors = ['red','blue','green']
#marker_styles = ['o','s','^']


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
    
    # Load verticies
    
    year = Years[i]
    file_name = f'Verticies'
    folder_name = f'NearOpt_Renew_Nuc_{year}'
    file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
    
    verticies = np.loadtxt(file_path, delimiter=',', dtype=int)
    
    # Load MAA sample (not that i use them)
    file_name = f'MAA_Samples'
    file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
    
    MAA_samples = np.loadtxt(file_path, delimiter=',', dtype=int)
    
    # Load Optimal mix
    file_name = f'OptimalSolution'
    file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
    
    Opt_Sol = np.loadtxt(file_path, delimiter=',', dtype=int)
    
    # Load Brownfield optimal mix
    file_name = f'BrownField_OptimalSolution'
    folder_name = f'BrownField_Nuc_{year}'
    file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\Myopic\{folder_name}\{file_name}'
   
    Brown_Opt_Sol = np.loadtxt(file_path, delimiter=',', dtype=int)
    
    
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
    
    
    reduction_percentage = (1 - Years[i]) * 100
    label = f'{reduction_percentage:.0f}% reduction'
    legend_labels.append(label)
    line = Line2D([0], [0], color=Colors[i], marker=marker_styles[i], markersize=8, label=label)
    legend_handles.append(line)
    # --------  Plot hull --------------------
    
    hull = ConvexHull(variable_verticies.values)
    
    # plot simplexes
    for simplex in hull.simplices:
        l0, = plt.plot(variable_verticies.values[simplex, 0], variable_verticies.values[simplex, 1], 'k-', 
                color = Colors[i], label = label,marker=marker_styles_lines[i],
                linewidth = 2, zorder = 0)
    
    
    # Also plot the optimal point
    
    plt.scatter(Opt_Sol[0],Opt_Sol[1],color=Colors[i],s=200 , 
                marker=marker_styles_dots[i],edgecolor='green',
                linewidths=5)
    
    # Also plot the Brownfield results
    plt.scatter(Brown_Opt_Sol[0],Brown_Opt_Sol[1],color=Colors[i],s=200 , 
                marker=marker_styles_dots[i],edgecolor='brown',
                linewidths=5)
    
    plt.xlabel('Nuclear capacity [MW]')
    plt.ylabel('Wind + Solar capacity [MW]')
    plt.title('Near optimal spaces - Greenfield')
    plt.xlim(-1000, 80000)
    
plt.legend(handles=legend_handles, labels=legend_labels, 
           loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=3)

plt.legend(['0% reduction','10% reduction','20% reduction','30% reduction',
            '40% reduction','60% reduction','78% reduction',
            '86% reduction','90% reduction'])



# I think i took the code below from Tim Toernes, and i have not used it yet

    #plt.show()
    # list of legend handles and labels
    #l_list, l_labels   = [l0, hb], ['Convex hull', 'Sample density']
    
    # if plot_MAA_points:
    #     # Plot vertices from solutions
    #     l1, = ax.plot(x, y,
    #               'o', label = "Near-optimal",
    #               color = 'lightcoral', zorder = 2)
    #     l_list.append(l1)
    #     l_labels.append('MAA points')
        
    # if show_text:
    #     ax.legend(l_list, l_labels, 
    #               loc = 'center', ncol = len(l_list),
    #               bbox_to_anchor = (0.5, -0.15), fancybox=False, shadow=False,)
    
    # Set limits
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    
    
    
    