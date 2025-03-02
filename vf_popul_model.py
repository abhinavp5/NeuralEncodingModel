import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
@Author: Abhinav - This is my main code for the population model, Radial and Spatial
"""
class VF_Population_Model:
    
    def __init__(self, vf_tip_size, aff_type, scaling_factor, density=None):
        self.sf = scaling_factor 
        self.vf_tip_size = vf_tip_size
        self.aff_type = aff_type
        self.results = None
        self.stress_data = None
        self.x_coords = None
        self.y_coords = None
        self.time_of_firing = None
        self.radial_stress_data = None
        self.radial_iff_data = None
        self.SA_radius = None
        self.g = None # g parameter from Merats paper
        self.h = None # h parameter from Merats paper
        self.density = density.lower().capitalize() if density else None

    def spatial_stress_vf_model(self, time_of_firing="peak", g=0.2, h=0.5):
        self.time_of_firing = time_of_firing
        self.g = g
        self.h = h

        coords = None
        stress_data = None
        if not self.density:
            coords = pd.read_csv(f"data/anika_new_data/{self.vf_tip_size}/{self.vf_tip_size}_spatial_coords_corr.csv")
        elif self.density in ["Low", "Med", "High", "Realistic"]:
            coords = pd.read_csv(f"data/anika_new_data/{self.density}/{self.vf_tip_size}/{self.vf_tip_size}_spatial_coords_corr_{self.density.lower()}.csv")
        else:
            logging.error(f"Density is not specified correctly: {self.density}")
            return

        self.x_coords = [float(row[0]) for row in coords.iloc[0:].values]
        self.y_coords = [float(row[1]) for row in coords.iloc[0:].values]

        if not self.density:
            stress_data = pd.read_csv(f"data/anika_new_data/{self.vf_tip_size}/{self.vf_tip_size}_spatial_stress_corr.csv")
        elif self.density in ["Low", "Med", "High", "Realistic"]:
            stress_data = pd.read_csv(f"data/anika_new_data/{self.density}/{self.vf_tip_size}/{self.vf_tip_size}_spatial_stress_corr_{self.density}.csv")
        else:
            logging.error(f"Density is not specified correctly: {self.density}")
            return

        time = stress_data['Time (ms)'].to_numpy()

        # Initialize lists for storing results
        afferent_type, x_pos, y_pos, spikes = [], [], [], []
        mean_firing_frequency, peak_firing_frequency = [], []
        first_spike_time, last_spike_time, stress_trace = [], [], []
        cumulative_mod_spike_times, entire_iff = [], []

        for i, row in coords.iterrows():
            i += 1  
            if f"Coord {i} Stress (kPa)" in stress_data.columns:
                stress = stress_data[f"Coord {i} Stress (kPa)"] * self.sf
            else:
                continue

            lmpars = lmpars_init_dict['t3f12v3final']
            if self.aff_type == "RA":
                lmpars['tau1'].value = 8
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1
                lmpars['k1'].value = 35
                lmpars['k2'].value = 0
                lmpars['k3'].value = 0.0
                lmpars['k4'].value = 0
            elif self.aff_type == "SA":
                lmpars['tau1'].value = 8
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1744.6
                lmpars['tau4'].value = np.inf
                lmpars['k1'].value = 0.74
                lmpars['k2'].value = 2.75
                lmpars['k3'].value = 0.07
                lmpars['k4'].value = 0.0312

            groups = MC_GROUPS
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=self.g, h=self.h)

            if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
                continue

            features, _ = pop_model(mod_spike_time, mod_fr_inst)

            afferent_type.append(self.aff_type)
            x_pos.append(row.iloc[0])
            y_pos.append(row.iloc[1])
            spikes.append(mod_spike_time)
            entire_iff.append(mod_fr_inst)
            mean_firing_frequency.append(features["Average Firing Rate"])

            peak_firing_frequency.append(np.max(mod_fr_inst) if time_of_firing == "peak" else 0)

            first_spike_time.append(mod_spike_time[0] if len(mod_spike_time) != 0 else None)
            last_spike_time.append(mod_spike_time[-1])
            stress_trace.append(stress)
            cumulative_mod_spike_times.append(mod_spike_time)

        self.results = {
            'afferent_type': self.aff_type,
            'x_position': x_pos,
            'y_position': y_pos,
            'spike_timings': spikes,
            'mean_firing_frequency': mean_firing_frequency,
            'peak_firing_frequency': peak_firing_frequency,
            'first_spike_time': first_spike_time,
            'last_spike_time': last_spike_time,
            'each_coord_stress': stress_trace,
            'entire_iff': entire_iff,
            'cumulative_mod_spike_times': cumulative_mod_spike_times
        }
        return self.results
