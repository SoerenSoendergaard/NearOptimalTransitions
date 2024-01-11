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

Years = [2020,2025,2028,2030,2032,2035,2040,2043,2045]
Years = [1,0.70,0.6,0.5,0.3,0.22,0.15,0.1,0.05, 0.02,0.01]

for i in range(len(Years)):
    
    if __name__ == '__main__':
        
        # Create or load network
        ############   Network setup ##############
        
        #network = 'network_2020.nc'
        year = Years[i]
        file_name = f'network_{year}.nc'
        file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\Myopic\Green\{file_name}'
        network = file_path
        
        ############# Continuing MAA ##############
        
        # Load options from configuration file
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
            
            
        # Set MAA variables to explore
        variables = {'Conventional': ['Generator', # Component type
                            ['Nuclear'], # Component carriers
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
                                          mga_slack = 0.1,
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
    
        
  
        # # Save verticies    
        year = Years[i]
        file_name = f'Verticies'
        folder_name = f'NearOpt_Renew_Nuc_{year}'
        file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
        # Create the folder if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savetxt(file_path, verticies, delimiter=',', fmt='%d')
        
        # Save MAA sample
        file_name = f'MAA_Samples'
        file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
        np.savetxt(file_path, MAA_samples, delimiter=',', fmt='%d')
        
        # Save opt_sol
        file_name = f'OptimalSolution'
        file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
        np.savetxt(file_path, opt_sol, delimiter=',', fmt='%d')
        
        # Save the optimum cost
        file_name = f'OptimalCost'
        file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\GreenSpaces\{folder_name}\{file_name}'
        obj_array = np.array([obj])
        np.savetxt(file_path, obj_array, delimiter=',', fmt='%d')
        