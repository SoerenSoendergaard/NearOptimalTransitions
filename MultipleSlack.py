# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:46:11 2023

@author: soere
"""

import PyMGA
from PyMGA.utilities.plot import near_optimal_space_2D, near_optimal_space_matrix
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

Slack = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]

#Slack = [0.07,0.08,0.09,0.1]

Year = 0.05 # This indicates which reduction case, the contours are added to.
            # As this process takes some time, i was chosen just to make the script
            # Process one near optimal space at a time

MainFolder = f'NearOpt_Renew_Nuc_MultiSlack'

for i in range(len(Slack)):
    
    if __name__ == '__main__':
        
        # Create or load network
        ############   Network setup ##############
        
        #network = 'network_2020.nc'
        
        file_name = f'network_{Year}.nc'
        file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\Myopic\Green\{file_name}'
        network = file_path
        
        ############# Continuing MAA ##############
        
        # Load options from configuration file
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
            
            
        # Set MAA variables to explore
        variables = {'Conventional': ['Generator', # Component type
                            ['Nuclear','NuclearWTes'], # Component carriers
                            'p_nom',], # Component variable to explore
                    'Wind+Solar': ['Generator',
                            ['Wind','Solar'],
                            'p_nom',],
                            } 
       
        #### PyMGA ####
        # PyMGA: Build case from PyPSA network
        case = PyMGA.cases.PyPSA_to_case(config, 
                                          network,
                                          variables = variables,
                                          mga_slack = Slack[i],
                                          n_snapshots = 8760)
        
        
        
        
        # PyMGA: Choose MAA method
        method = PyMGA.methods.MAA(case)
        
        # PyMGA: Solve optimal system
        opt_sol, obj, n_solved = method.find_optimum()
        
        # Draw optimal system (optional)
        # draw_network(n_solved, show_capacities = True)
        
        # PyMGA: Search near-optimal space using chosen method
        verticies, directions, _, _ = method.search_directions(14, n_workers = 16)
    
        # PyMGA: Sample the identified near-optimal space
        MAA_samples = PyMGA.sampler.har_sample(100_000, x0 = np.zeros(len(variables.keys())), 
                                                directions = directions, 
                                                verticies = verticies)
    
        
  
        # Save verticies    
        slack = Slack[i]
        file_name = f'Verticies'
        folder_name = f'NearOpt_Renew_Nuc_{slack}_{Year}'
        file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\NearOpt_Renew_Nuc_MultiSlack\{folder_name}\{file_name}'
        # Create the folder if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savetxt(file_path, verticies, delimiter=',', fmt='%d')
        
        
        # All the sampled points of the spaces could be saved like this (it has not been needed yet):
        # # Save MAA sample
        # file_name = f'MAA_Samples'
        # file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
        # np.savetxt(file_path, MAA_samples, delimiter=',', fmt='%d')
        
        # # Save opt_sol
        # file_name = f'OptimalSolution'
        # file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
        # np.savetxt(file_path, opt_sol, delimiter=',', fmt='%d')
        
    
