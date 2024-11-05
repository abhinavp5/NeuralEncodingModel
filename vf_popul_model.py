import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from lmfit import minimize, fit_report, Parameters
from aim2_population_model_spatial_aff_parallel import get_mod_spike
from model_constants import (MC_GROUPS, LifConstants)
from popul_model import pop_model


#Global Variables
lmpars_init_dict = {}
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=1744.6, vary=False)
lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=.74, vary=False, min=0) #a constant
lmpars.add('k2', value=2.75, vary=False, min=0) #b constant
lmpars.add('k3', value=.07, vary=False, min=0) #c constant
lmpars.add('k4', value=.0312, vary=False, min=0)
lmpars_init_dict['t3f12v3final'] = lmpars

class VF_Population_Model:
    
    def __init__(self, vf_tip_size, aff_type):
        self.vf_tip_size = vf_tip_size
        self.aff_type = aff_type
        self.results = None
        self.stress_data = None
    """
        functino takes in a vf_tip_size (given that that there is data assicated with it) an
        afferent type, and runs the single unit model for all of those coordinates with the data
        
    """
    def spatial_stress_vf_model(self, scaling_factor = 0.3):

        #reading data in 
        coords = pd.read_csv(f"data/vfspatial/{self.vf_tip_size}_spatial_coords.csv", header = None)
        x_coords = coords[0]
        y_coords = coords[1]
        stress_data = pd.read_csv(f"data/vfspatial/3.61_spatial_stress.csv", )
        time = stress_data['Time (ms)'].to_numpy()

        afferent_type = []
        x_pos = []
        y_pos = []
        spikes = []
        mean_firing_frequency = []
        peak_firing_frequency = []
        first_spike_time = []
        last_spike_time = []

        #iterating through each of the coordinates
        for i, row in coords.iloc[1:].iterrows():
            #getting stress data
            
            if f"Coord {i} Stress (kPa)" in stress_data.columns:
                stress = stress_data[f"Coord {i} Stress (kPa)"]
            else:
                logging.warning("STRESS VALUE COULD NOT BE INDEXED")

            lmpars = lmpars_init_dict['t3f12v3final']
            if self.aff_type == "RA":
                lmpars['tau1'].value = 2.5
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1
                lmpars['k1'].value = 35
                lmpars['k2'].value = 0
                lmpars['k3'].value = 0.0
                lmpars['k4'].value = 0

            groups = MC_GROUPS
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress)

            if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
                logging.warning(f"SPIKES COULD NOT BE GENERATED FOR {self.vf_tip_size}")
                continue

            if len(mod_spike_time) != len(mod_fr_inst):
                if len(mod_fr_inst) > 1:
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
                else:
                    mod_fr_inst_interp = np.zeros_like(mod_spike_time)
            else:
                mod_fr_inst_interp = mod_fr_inst

            features, _ = pop_model(mod_spike_time,mod_fr_inst_interp)

            #appending stuff to lists
            afferent_type.append(self.aff_type)
            x_pos.append(row[0])
            y_pos.append(row[1])
            spikes.append(len(mod_spike_time) if len(mod_spike_time) !=0 else None)
            mean_firing_frequency.append(features["Average Firing Rate"])
            peak_firing_frequency.append(np.max(mod_fr_inst_interp))
            first_spike_time.append(mod_spike_time[0] if len(mod_spike_time) != None else None)
            last_spike_time.append(mod_spike_time[-1])

            
        model_results = {
            'afferent_type': self.aff_type,
            'x_position': x_pos,
            'y_position': y_pos,
            'num_of_spikes' : spikes,
            'mean_firing_frequency' : mean_firing_frequency,
            'peak_firing_frequency' : peak_firing_frequency, 
            'first_spike_time': first_spike_time,
            'last_spike_time' : last_spike_time
        }

        self.results = model_results
        return model_results

    def radial_stress_vf_model(self,scaling_factor = 0.3):
        """ Read in the Radial which has sample stress traces for every 2mm from a center point
        to calculate firing"""

        #regex pattern for exstracting the distance from the middle point
        distance_regex = r'\d\.\d{2}'

        #reading data from spatial stress data file for 50 (or n) data points in a grid
        coords = pd.read_csv(f"data/vfspatial/{self.vf_tip_size}_spatial_coords.csv", header = None)
        stress_df = pd.read_csv(f"data/vfspatial/{self.vf_tip_size}_spatial_stress.csv", )
        time = stress_df['Time (ms)'].to_numpy()

        #Reading in the radial stress file
        radial_stress = pd.read_csv(f"data/vfspatial/{self.vf_tip_size}_radial_stress.csv")
        radial_time = stress_df['Time (ms)'].to_numpy()
        
        stress_data = {}
        iff_data = {}


        #Outer loop top iterate through all n spatial points
        for i, row in coords.iloc[1:].iterrows():
            radial_spatial_flag = True
            stress_data[i] = {}
            iff_data[i] = {}

            if f"Coord {i} Stress (kPa)" in stress_df.columns:
                spatial_stress = stress_df[f"Coord {i} Stress (kPa)"]
                spatial_stress_max = np.max(spatial_stress)

                # Inner loop to iterate through radial distances
                for col in radial_stress.columns[1:]:
                    distance_from_center = float(re.findall(distance_regex, col)[0])

                    # Initialize lists for each coordinate-distance pair
                    afferent_type = []
                    x_pos = []
                    y_pos = []
                    spikes = []
                    mean_firing_frequency = []
                    peak_firing_frequency = []
                    first_spike_time = []
                    last_spike_time = []

                    if radial_spatial_flag:
                        radial_stress_vals = radial_stress[col]
                        radial_stress_max = np.max(radial_stress_vals)
                        distance_scaling_factor = spatial_stress_max / radial_stress_max
                        radial_spatial_flag = False
                    
                    scaled_stress = radial_stress[col] * distance_scaling_factor * scaling_factor

                    stress_data[i][distance_from_center] = {
                        "Time": radial_time,
                        distance_from_center: scaled_stress.to_numpy()
                    }

                    lmpars = lmpars_init_dict['t3f12v3final']
                    if self.aff_type == "RA":
                        lmpars['tau1'].value = 2.5
                        lmpars['tau2'].value = 200
                        lmpars['tau3'].value = 1
                        lmpars['k1'].value = 35
                        lmpars['k2'].value = 0
                        lmpars['k3'].value = 0.0
                        lmpars['k4'].value = 0

                    groups = MC_GROUPS
                    mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, stress_data[i][distance_from_center]["Time"], stress_data[i][distance_from_center][distance_from_center])

                    if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
                        logging.warning(f"SPIKES COULD NOT BE GENERATED FOR {self.vf_tip_size}")
                        continue

                    if len(mod_spike_time) != len(mod_fr_inst):
                        if len(mod_fr_inst) > 1:
                            mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
                        else:
                            mod_fr_inst_interp = np.zeros_like(mod_spike_time)
                    else:
                        mod_fr_inst_interp = mod_fr_inst

                    features, _ = pop_model(mod_spike_time, mod_fr_inst_interp)

                    # Append single values to the lists
                    afferent_type.append(self.aff_type)
                    x_pos.append(row[0])
                    y_pos.append(row[1])
                    spikes.append(len(mod_spike_time) if len(mod_spike_time) != 0 else None)
                    mean_firing_frequency.append(features["Average Firing Rate"])
                    peak_firing_frequency.append(np.max(mod_fr_inst_interp))
                    first_spike_time.append(mod_spike_time[0] if len(mod_spike_time) != 0 else None)
                    last_spike_time.append(mod_spike_time[-1])

                    # Store each coordinate-distance dictionary within iff_data
                    iff_data[i][distance_from_center] = {
                        'afferent_type': self.aff_type,
                        'x_position': x_pos[0],
                        'y_position': y_pos[0],
                        'num_of_spikes': spikes[0],
                        'mean_firing_frequency': mean_firing_frequency[0],
                        'peak_firing_frequency': peak_firing_frequency[0],
                        'first_spike_time': first_spike_time[0],
                        'last_spike_time': last_spike_time[0]
                    }
            else:
                logging.warning("STRESS VALUE COULD NOT BE INDEXED")
        self.stress_data = stress_data
        self.results = iff_data

        

    def aggregate_results(self):
        df = pd.DataFrame(self.results)
        file_path = f"generated_csv_files/{self.vf_tip_size}_vf_popul_model.csv"
        df.to_csv(file_path, index = False)
        return file_path 
    
    def plot_spatial_coords(self):
        """
        Plots the iffs on a grid for the original n points, the magniude of the peak firing
        frequency directly affects the size of the circle plotted, and the opacity
        """
        #colors for differnet afferents
        colors = {'SA': '#31a354', 'RA': '#3182bd'}
        plt.figure(figsize=(10, 5))

        # Plot the stimulus locations as circles
        x_positions = self.results.get("x_position")
        y_positions = self.results.get("y_position")
        mean_iffs = self.results.get("mean_firing_frequency")
        peak_iffs = self.results.get("peak_firing_frequency")
        
        x_positions = [float(value) for value in x_positions]
        y_positions = [float(value) for value in y_positions]
        #scaling peak_iffs so it looks better when plotting
    
        alphas = [float(value)/max(peak_iffs) for value in peak_iffs]

        #Scatter plot
                # Plot the stimulus locations as circles
        for x_pos, y_pos, radius, alpha in zip(x_positions, y_positions, peak_iffs, alphas):
            plt.gca().add_patch(
                patches.Circle((x_pos, y_pos), radius*2, edgecolor='black', facecolor = colors.get(self.aff_type) , linewidth=1, alpha = 0.5)
            )
        plt.xlabel('Length (mm)')
        plt.ylabel('Width (mm)')
        plt.title("VF Afferent Stress Distribution")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(min(x_positions) - 1, max(x_positions) + 1)
        plt.ylim(min(y_positions) - 1, max(y_positions) + 1)
        plt.savefig(f"vf_graphs/aggregated_results_on_grid/{self.vf_tip_size}_{self.aff_type}_constant_opacity.png")

        

    def get_iffs(self):
        return self.results
    
    def get_stress_traces(self):
        return self.stress_data

if __name__ == '__main__':
    #creates model class
    vf_model = VF_Population_Model(4.17, "SA")

    #runs the model which calculates the results
    # vf_model.spatial_stress_vf_model()
    vf_model.radial_stress_vf_model()
    iffs_results = vf_model.get_iffs()
    stress_results = vf_model.get_stress_traces()
    print(stress_results)

    # #writing the data to csv
    # vf_model.aggregate_results()

    # #plotting the aggregated_results
    # vf_model.plot()

