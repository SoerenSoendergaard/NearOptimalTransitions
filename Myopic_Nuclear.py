# -*- coding: utf-8 -*-

"""
Created on Thu Aug 24 10:54:54 2023

@author: soere
"""


import statistics 
import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
import dill
import xarray as xr
import os
warnings.filterwarnings("ignore")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def annuity(n,r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n

# Setup system

Myopic = "Off"
TES = 'Off'


Network = pypsa.Network()

# Define the hours of the analysis
FirstTime = "2015-01-01T00:00:00Z"
LastTime = "2015-12-31T23:00:00Z"

Hours_2015 = pd.date_range(FirstTime,LastTime, freq='H')
Network.set_snapshots(Hours_2015)

Network.add("Bus",'DEU EL')

Discount_rate = 0.07 # For all technologies

#%% Define technologies

# Everything is in dollar / MW

# Tecnologies to define:
    # Natural gas
    # Natural gas with carbon capture
    # Solar
    # Wind

# Defining Natural gas

Network.add("Carrier", "Natural gas",co2_emissions = 0.49)
Lifetime = 30
investment = 954*1000
FOM = 12.15*1000
fuel = 19.11
efficiency = 0.53  # From their github-excel
VOM = 1.86

Co2_int = 0.2

Capex = (annuity(Lifetime,Discount_rate)*(investment+FOM))
#Marg = (fuel/efficiency)+VOM
Marg = 20.97
Capex = 10.16*8760
Network.add("Generator",
        "Natural gas",
        bus="DEU EL",
        p_nom_extendable=True,
        carrier="Natural gas",
        #p_min_pu = MinLoad,
        capital_cost = Capex,
        marginal_cost = Marg)


# Defining Natural gas with Co2-capture

Network.add("Carrier", "Natural gas CCS",co2_emissions = 0.17)
Lifetime = 30
investment = 3569*1000
FOM = 27.48*1000
fuel = 21.37
efficiency = 0.48 # From their github-excel
VOM = 5.82


Capex = (annuity(Lifetime,Discount_rate)*(investment+FOM))
Capex = 26.77*8760
#Marg = (fuel/efficiency)+VOM
Marg = 27.19
Network.add("Generator",
        "Natural gas CCS",
        bus="DEU EL",
        p_nom_extendable=True,
        carrier="Natural gas CCS",
        #p_min_pu = MinLoad,
        capital_cost = Capex,
        marginal_cost = Marg)


# Define solar

Network.add("Carrier", "Solar")

Lifetime = 25
investment = 1331*1000
FOM = 15.19*1000

Capex = (annuity(Lifetime,Discount_rate)*(investment+FOM))
Marg = 0


#data = pd.read_csv (r'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\pv_optimal.csv')
data = pd.read_csv (r'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\20201218_DE_mthd3_solar.csv')
year_2019_data = data[data['year'] == 2019]
Caps = year_2019_data['s_cfs']

# idx_1 = float(data[data["utc_time"]==FirstTime].index.values)
# idx_2 = float(data[data["utc_time"]==LastTime].index.values)
# data = data["DEU"]

# Caps = data.loc[idx_1:idx_2]
Caps.index = Hours_2015

Capex = 14.77*8760
Network.add("Generator",
        "Solar",
        bus="DEU EL",
        p_nom_extendable=True,
        carrier="Solar",
        capital_cost = Capex,
        marginal_cost = Marg,
        p_max_pu = Caps)


# Define wind

Network.add("Carrier", "Wind")

Lifetime = 25
investment = 1319*1000
FOM = 26.22*1000

Capex = (annuity(Lifetime,Discount_rate)*(investment+FOM))
Marg = 0

# data = pd.read_csv (r'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\offshore_wind_1979-2017.csv') 
# idx_1 = float(data[data["utc_time"]==FirstTime].index.values)
# idx_2 = float(data[data["utc_time"]==LastTime].index.values)
# data = data["DEU"]

data = pd.read_csv (r'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\20201218_DE_mthd3_wind.csv')
year_2019_data = data[data['year'] == 2019]
Caps = year_2019_data['w_cfs']


#MultiplyFactor = 1

#Caps = data.loc[idx_1:idx_2]*MultiplyFactor
#TF = Caps >= 1
#Caps[TF] = 1
Caps.index = Hours_2015
Capex = 15.91*8760
Network.add("Generator",
        "Wind",
        bus="DEU EL",
        p_nom_extendable=True,
        carrier="Wind",
        capital_cost = Capex,
        marginal_cost = Marg,
        p_max_pu = Caps)


# Define cheap nuclear

Network.add("Carrier", "Nuclear")
#Network.add("Bus",'Nuclear', carrier = "Nuclear")
Lifetime = 40
investment = (4000-1854)*1000
FOM = 121.13*1000

Capex = (annuity(Lifetime,Discount_rate)*(investment+FOM))

investment = 1854*1000
Lifetime = 30
FOM = 33.6
Capex = (annuity(Lifetime,Discount_rate)*(investment+FOM))+Capex

Marg = 0

### Using their numbers to hit capex of 4000

#Capex = (13.472222 + 20.905199)*8760
Capex = 77.08*8760


Network.add("Generator",
        "Nuclear",
        bus="DEU EL",
        p_nom_extendable=True,
        carrier="Nuclear",
        #p_min_pu = MinLoad,
        efficiency = 0.37,
        capital_cost = Capex,
        marginal_cost = Marg)

# Define cheap nuclear with TES

if TES == 'On':
    Network.add("Carrier", "NuclearWTes")
    Network.add("Bus",'NuclearWTes', carrier = "NuclearWTes")
    
    ### The reactor
    #Capex = 77.08*8760
    Capex = 20.83*8760
    Network.add("Generator",
            "NuclearWTes",
            bus="NuclearWTes",
            p_nom_extendable=True,
            carrier="NuclearWTes",
            #p_min_pu = MinLoad,
            efficiency = 1,
            capital_cost = Capex,
            marginal_cost = 0)
    
    # Adding TES storage
    
    Network.add("Carrier","TES")
    Network.add("Bus",'TES', carrier = "TES")
    
    Lifetime = 30
    investment = 74.47*1000
    FOM = 2.23*1000
    
    Capex = (annuity(Lifetime,Discount_rate)*(investment+FOM))
    Capex = 2.23*8760
    Network.add("Store",
      "TES",
      bus = "TES",
      e_nom_extendable = True,
      e_cyclic = True,
      #standing_loss = 0.00026,
      capital_cost = Capex)
    
    # Add link between nuclear and TES
    Network.add("Link",
          "TES - NuclearWTes",
          bus0 = "NuclearWTes",
          bus1 = "TES",
          p_nom_extendable = True,
          efficiency = 1,
          capital_cost = 0)
    
    # Add link between TES and generator
    Network.add("Carrier","Generator")
    Network.add("Bus",'Generator', carrier = "Generator")
    Capex = 20.91*8760
    Network.add("Link",
          "TES - Generator",
          bus0 = "TES",
          bus1 = "Generator",
          p_nom_extendable = True,
          efficiency = 0.37,
          capital_cost = Capex)
    
    # Add link between TES and EL
    Network.add("Link",
          "Generator - EL",
          bus0 = "Generator",
          bus1 = "DEU EL",
          p_nom_extendable = True,
          efficiency = 1,
          capital_cost = 0)



# Add battery storage

Network.add("Carrier","Battery")
Network.add("Bus",'Battery', carrier = "Battery")

investment = 365.77*1000
FOM = 12.32*1000
Capex = (annuity(Lifetime,Discount_rate)*(investment+FOM))
Capex = 7.35*8760
Network.add("Store",
  "Battery",
  bus = "Battery",
  e_nom_extendable = True,
  e_cyclic = True,
  capital_cost = Capex) # This is prettier



# Add link from betteries to electricity grid through inverter
Network.add("Link",
      "Battery",
      bus0 = "Battery",
      bus1 = "DEU EL",
      p_nom_extendable = True,
      p_min_pu = -1,
      efficiency = 1,
      #standing_loss = 0.00001,
      capital_cost = 0)


#%% add load

#data = pd.read_csv (r'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Electricity_Demand.csv')
data = pd.read_csv(r'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\electricity_demand.csv',sep=';',index_col=[0],header=0)
#idx_1 = float(data[data["utc_time"]==FirstTime].index.values)
#idx_2 = float(data[data["utc_time"]==LastTime].index.values)
Demand = data["DEU"]

#Demand = data.loc[idx_1:idx_2]
Demand.index = Hours_2015

Network.add("Load",
        "load", 
        bus="DEU EL",
        p_set=Demand)


#%% Setup Co2 analysis
Technologies = ["Natural gas","Natural gas CCS","Solar","Wind","Nuclear","NuclearWTes","Battery"]
#Generators = ["Natural gas","Natural gas CCS","Solar","Wind","Nuclear","NuclearWTes"]
Generators = ["Natural gas","Natural gas CCS","Solar","Wind","Nuclear"]
Storage = ["Battery"]
NrTechs = len(Generators)+len(Storage)
LifeTimes = [30,30,25,25,40,40,10]
Co2_Limits = [1,0.90,0.8,0.70,0.6,0.4,0.22,0.135,0.1,0.05, 0.02]
Years = [2020,2025,2028,2030,2032,2035,2040,2043,2045,2048, 2050]

Co2_Limits = [1,0.90,0.8,0.70,0.6,0.5,0.4,0.3,0.22,0.15,0.135,0.1,0.08,0.05, 0.02,0.01]
Years = [2020,2025,2028,2030,2032,2033,2035,2037,2040,2041,2043,2045,2047,2048, 2050,2055]

#Co2_Limits = [1,0.1,0.05]
#Years = [2020,2045,2050]


# Initially solve year 1, which is then used to define CO2 constraint

Network.lopf(Network.snapshots,pyomo=False,solver_name='gurobi')


###########  If the CO2 constraint should be defined from historic measures ###########

# Part_of_emmisions = 0.17 # Part of 1990 emmisions from power production is 39, but
#                          # but this is both from el and heat, so half?
# Co2_1990 = 1128090000*Part_of_emmisions # tons of Co2 equivalent

##################################################################

Co2_1990 = sum((Network.snapshot_weightings.generators @ Network.generators_t.p) / Network.generators.efficiency * Network.generators.carrier.map(Network.carriers.co2_emissions))


# Scheme for dumping results
zero = np.zeros((len(Co2_Limits)))
#d = {"Natural gas":zero,"Natural gas CCS":zero,"Solar":zero,"Wind":zero,"Nuclear":zero,"NuclearWTes":zero,"Battery":zero}
d = {"Natural gas":zero,"Natural gas CCS":zero,"Solar":zero,"Wind":zero,"Nuclear":zero,"Battery":zero}


Capacities = pd.DataFrame(data=d)
Production = pd.DataFrame(data=d)
PartOfEURPerWatt = pd.DataFrame(data=d)

Co2_price = []

Co2_Limits_To_Save = [1,0.70,0.6,0.5,0.3,0.22,0.15,0.1,0.05, 0.02,0.01]

for i in range(len(Years)):
    # Define Co2 constraint:
        co2_limit = Co2_1990*Co2_Limits[i]
        Network.add("GlobalConstraint",
                    "co2_limit",
                    type="primary_energy",
                    carrier_attribute="co2_emissions",
                    sense="<=",
                    constant=co2_limit)
        Network.lopf(Network.snapshots,pyomo=False,solver_name='gurobi')
        
        year = Co2_Limits[i]  # Origionally i named things after the year of the transition
             
        # NuclearCapacity = Network.generators.p_nom_opt['Nuclear'] + Network.generators.p_nom_opt['NuclearWTes']
        NuclearCapacity = Network.generators.p_nom_opt['Nuclear']
        WindSolar = Network.generators.p_nom_opt['Wind'] + Network.generators.p_nom_opt['Solar']
        opt_sol = [NuclearCapacity,WindSolar]
        
        if  Co2_Limits[i] in Co2_Limits_To_Save:
        
            if Myopic == "On":
            
                # Save opt_sol
                file_name = f'BrownField_OptimalSolution'
                folder_name = f'BrownField_Nuc_{year}'
                file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\Myopic\{folder_name}\{file_name}'
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                np.savetxt(file_path, opt_sol, delimiter=',', fmt='%d')
                
                # Save the network
                file_name = f'network_{year}.nc'
                file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\Myopic\{file_name}'
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                Network.export_to_netcdf("file_name.nc")
                
            if Myopic == "Off":
                
                # Save the network
                file_name = f'network_{year}.nc'
                file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\Myopic\Green\{file_name}'
                
                # Define the desired directory path
                desired_directory = r"C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\Myopic\Green"
    
                # Change the current working directory to the desired directory
                os.chdir(desired_directory)
                Network.export_to_netcdf(file_name)
                
                file_name = f'GreenField_OptimalSolution'
                folder_name = f'GreenField_Nuc_{year}'
                file_path = rf'C:\Users\soere\OneDrive\UNI\Kandidat\Semester 1\Renewable energy systems\Pre_masters\Results\Myopic\{folder_name}\{file_name}'
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                np.savetxt(file_path, opt_sol, delimiter=',', fmt='%d')
            
        
        # Dump network data for plotting
        for j in range(len(Generators)):
            PriceOfTech = Network.generators.capital_cost[Generators[j]]*Network.generators.p_nom_opt[Generators[j]] + sum(Network.generators.marginal_cost[Generators[j]]*Network.generators_t.p[Generators[j]])
            Production.iloc[i,j] = Network.generators_t.p[Generators[j]].sum()
            Capacities.iloc[i,j] = Network.generators.p_nom_opt[Generators[j]]
            PartOfEURPerWatt.iloc[i,j] = PriceOfTech/Network.loads_t.p.sum()
        
        Battery_price = Network.stores.capital_cost["Battery"]*Network.stores.e_nom_opt["Battery"]
        PartOfEURPerWatt.iloc[i, PartOfEURPerWatt.columns.get_loc("Battery")] = Battery_price/Network.loads_t.p.sum()
        Capacities.iloc[i,Capacities.columns.get_loc("Battery")] = Network.stores.e_nom_opt["Battery"]  
        
        if i == 0:
            # For the first solution, mark existing tecnologies
            # only theese should decay
            M = Network.generators
            TF = M.p_nom_opt != 0
            TF = TF.tolist()
            
            Atributing_technologies = [generator for generator, is_true in zip(Generators, TF) if is_true]
            
        
        if Myopic == "On":
            # Update capacities
            Capacities_4_update = Network.generators.p_nom_opt
            
            TimeStep = Years[i+1]-Years[i]
    
            for i in range(len(Generators)):
                
                if TF[i] == True: # Let origional technologies decay linearly
                    # Account for decay
                    Capacities_4_update[i] = Capacities_4_update[i]-(Capacities_4_update[i]*(TimeStep/LifeTimes[i]))
                    
                Network.generators.p_nom_min[Generators[i]] = Capacities_4_update[i]
                
        # Remove the current Co2 constraint
        
        Co2_price.append(Network.global_constraints.mu) 
        Network.remove("GlobalConstraint","co2_limit")
    


Co2_Limits = [x for x in Co2_Limits if x > 0]
fig,ax = plt.subplots()
plt.stackplot(Co2_Limits,PartOfEURPerWatt.iloc[:,0],
              PartOfEURPerWatt.iloc[:,1],
              PartOfEURPerWatt.iloc[:,2],
              PartOfEURPerWatt.iloc[:,3],
              PartOfEURPerWatt.iloc[:,4],
              PartOfEURPerWatt.iloc[:,5],
              labels = ["Natural gas","Natural gas CCS","Solar","Wind","Nuclear","Battery"], 
              colors = ["black","grey","yellow","blue","brown","pink"])



######## If tes where to be included   ##################

# Trying to do the plot with label:
# fig,ax = plt.subplots()
# plt.stackplot(Co2_Limits,PartOfEURPerWatt.iloc[:,0],
#               PartOfEURPerWatt.iloc[:,1],
#               PartOfEURPerWatt.iloc[:,2],
#               PartOfEURPerWatt.iloc[:,3],
#               PartOfEURPerWatt.iloc[:,4],
#               PartOfEURPerWatt.iloc[:,5],
#               PartOfEURPerWatt.iloc[:,6],
#               labels = ["Natural gas","Natural gas CCS","Solar","Wind","Nuclear","NuclearWTes","Battery"], 
#               colors = ["black","grey","yellow","blue","brown","purple","pink"])

##########################################################



plt.legend(fancybox=True, loc='upper center', bbox_to_anchor=(0.5, -0.22),shadow=True, ncol=4)
plt.xlim(1, 0)
new_ticks = [1, 0.8, 0.6, 0.5, 0.35, 0.2, 0.1, 0.0]
plt.xticks(new_ticks)


plt.xlabel("Percentage reduction in Co2",fontsize=14)
plt.ylabel("Dollar/MWh",fontsize=14)


new_ticks = [1,0.90,0.8,0.70,0.6,0.5,0.4,0.3,0.22,0.15,0.08,0.05,0.01]
new_ticks_labels = np.subtract(1,new_ticks)*100
new_ticks_labels = np.round(new_ticks_labels, decimals=1)
plt.xticks(new_ticks, rotation=90)
ax.set_xticklabels(new_ticks_labels, rotation = 90)


if Myopic == "On":    
    plt.title('System cost as function of reduction (brownfield)\n EIA Nuclear cost',fontsize=18)
else:
    plt.title('System cost as function of reduction (greenfield)\n EIA Nuclear cost',fontsize=18)

              




