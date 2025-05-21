import numpy as np
import pandas as pd
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import math
from aim2_population_model_spatial_aff_parallel import get_mod_spike
from model_constants import (MC_GROUPS, LifConstants)
import os
from lmfit import Parameters
from popul_model import pop_model


# Global Variables
lmpars_init_dict = {}
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False) #tauRI(ms)
lmpars.add('tau2', value=200, vary=False) #tauSI(ms)
lmpars.add('tau3', value=1744.6, vary=False)#tauUSI(ms)
lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=.74, vary=False, min=0) #a constant
lmpars.add('k2', value=.2088, vary=False, min=0) #b constant
lmpars.add('k3', value=.07, vary=False, min=0) #c constant
lmpars.add('k4', value=.0312, vary=False, min=0)
lmpars_init_dict['t3f12v3final'] = lmpars

class VF_Population_Model:
    """
    A class to model the population response of afferents to vibratory force (VF) stimuli.
    It uses spatial and radial stress data to simulate afferent firing responses.
    """

    def __init__(self, vf_tip_size, aff_type, scaling_factor, density=None):
        """
        Initialize the VF Population Model.

        Parameters:
        - vf_tip_size (float): The size of the VF tip in mm.
        - aff_type (str): The type of afferent ("SA" or "RA").
        - scaling_factor (float): The scaling factor applied to stress data.
        - density (str, optional): The density of afferents ("Low", "Med", "High", "Realistic").
        """
        self.sf = scaling_factor
        self.vf_tip_size = vf_tip_size
        self.aff_type = aff_type
        self.density = density.lower().capitalize() if density else None

        # Instance variables for storing data
        self.results = None
        self.stress_data = None
        self.x_coords = None
        self.y_coords = None
        self.time_of_firing = None
        self.radial_stress_data = None
        self.radial_iff_data = None
        self.SA_radius = None
        self.g = None
        self.h = None

    def spatial_stress_vf_model(self, time_of_firing="peak", g=0.2, h=0.5):
        """
        Computes the stress model based on spatial coordinates and firing times.

        Parameters:
        - time_of_firing (str or float): "peak" or a specific firing time in ms.
        - g (float): Model parameter for spike generation.
        - h (float): Model parameter for spike generation.

        Returns:
        - dict: A dictionary containing afferent responses and spike information.
        """
        self.time_of_firing = time_of_firing
        self.g = g
        self.h = h

        # Load spatial coordinate data
        coords_file = f"data/P2/{self.density}/{self.vf_tip_size}/{self.vf_tip_size}_spatial_coords_corr_{self.density.lower()}.csv"
        coords = pd.read_csv(coords_file)

        self.x_coords = coords.iloc[:, 0].astype(float).tolist()
        self.y_coords = coords.iloc[:, 1].astype(float).tolist()

        # Load stress data
        stress_file = f"data/P2/{self.density}/{self.vf_tip_size}/{self.vf_tip_size}_spatial_stress_corr_{self.density.lower()}.csv"
        stress_data = pd.read_csv(stress_file)
        time = stress_data['Time (ms)'].to_numpy()

        # Initialize data structures
        model_results = {
            "afferent_type": self.aff_type,
            "x_position": [],
            "y_position": [],
            "spike_timings": [],
            "mean_firing_frequency": [],
            "peak_firing_frequency": [],
            "first_spike_time": [],
            "last_spike_time": [],
            "each_coord_stress": [],
            "entire_iff": [],
            "cumulative_mod_spike_times": []
        }

        # Process each coordinate
        for i, row in coords.iterrows():
            stress_col = f"Coord {i+1} Stress (kPa)"
            if stress_col not in stress_data.columns:
                continue  # Skip if stress data is missing

            stress = stress_data[stress_col] * self.sf

            # Compute spikes using model
            lmpars = lmpars_init_dict['t3f12v3final']
            if self.aff_type == "RA":
                lmpars['tau1'].value = 8
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1
                lmpars['k1'].value = 35
                lmpars['k2'].value = 0
                lmpars['k3'].value = 0.0
                lmpars['k4'].value = 0

            groups = MC_GROUPS
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=self.g, h=self.h)

            if len(mod_spike_time) == 0:
                continue  # Skip if no spikes

            if len(mod_spike_time) != len(mod_fr_inst):
                if len(mod_fr_inst) > 1:
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
                else:
                    mod_fr_inst_interp = np.zeros_like(mod_spike_time)
            else:
                mod_fr_inst_interp = mod_fr_inst

            features, _ = pop_model(mod_spike_time, mod_fr_inst_interp)

            # Store results
            model_results["x_position"].append(row.iloc[0])
            model_results["y_position"].append(row.iloc[1])
            model_results["spike_timings"].append(mod_spike_time.tolist())
            model_results["mean_firing_frequency"].append(features["Average Firing Rate"])
            
            if time_of_firing == "peak":
                peak_fr = np.max(mod_fr_inst_interp)
                peak_fr_idx = np.argmax(mod_fr_inst_interp)
                peak_fr_time = mod_spike_time[peak_fr_idx] if peak_fr_idx < len(mod_spike_time) else None
                model_results["peak_firing_frequency"].append(peak_fr)
            else:
                difference_array = time_of_firing - np.array(mod_spike_time)
                positive_indices = np.where(difference_array > 0)[0]
                if len(positive_indices) > 0:
                    closest_spike_idx = positive_indices[np.argmin(difference_array[positive_indices])]
                    temp_fr_inst_interp = mod_fr_inst_interp[closest_spike_idx]
                else:
                    temp_fr_inst_interp = 0
                model_results["peak_firing_frequency"].append(temp_fr_inst_interp)

            model_results["first_spike_time"].append(mod_spike_time[0])
            model_results["last_spike_time"].append(mod_spike_time[-1])
            model_results["each_coord_stress"].append(stress.tolist())
            model_results["entire_iff"].append(mod_fr_inst_interp.tolist())
            model_results["cumulative_mod_spike_times"].append(mod_spike_time.tolist())

        self.results = model_results
        return model_results

    def radial_stress_vf_model(self, g=0.2, h=0.5):
        """
        Computes the stress model based on radial distances from the center.

        Parameters:
        - g (float): Model parameter for spike generation.
        - h (float): Model parameter for spike generation.
        """
        self.g = g
        self.h = h

        # Load radial stress data
        if self.density == "Realistic":
            if self.vf_tip_size == 4.56:
                radial_stress_file = f"data/P3/{self.density}/{self.vf_tip_size}/{self.vf_tip_size}_radial_stress_corr_{self.density.lower()}.csv"
            else:
                radial_stress_file = f"data/P4/{self.density}/{self.vf_tip_size}/{self.vf_tip_size}_radial_stress_corr_{self.density.lower()}.csv"
        else:
            radial_stress_file = f"data/P2/{self.density}/{self.vf_tip_size}/{self.vf_tip_size}_radial_stress_corr_{self.density.lower()}.csv"
        
        logging.info(f"Attempting to read file: {radial_stress_file}")
        
        try:
            radial_stress = pd.read_csv(radial_stress_file)
            time_col = 'Time (ms)' if 'Time (ms)' in radial_stress.columns else 'Time'
            radial_time = radial_stress[time_col].to_numpy()
            
            stress_data = {}
            iff_data = {}

            # Process each radial distance
            for col in radial_stress.columns:
                if col == time_col:
                    continue
                    
                matches = re.findall(r'\d\.\d{2}', col)
                if not matches:
                    continue

                distance_from_center = float(matches[0])
                scaled_stress = radial_stress[col] * self.sf
                stress_data[distance_from_center] = {
                    "Time": radial_time,
                    "Stress": scaled_stress.to_numpy()
                }

                # Compute spikes using model
                lmpars = lmpars_init_dict['t3f12v3final']
                if self.aff_type == "RA":
                    lmpars['tau1'].value = 8
                    lmpars['tau2'].value = 200
                    lmpars['tau3'].value = 1
                    lmpars['k1'].value = 35
                    lmpars['k2'].value = 0
                    lmpars['k3'].value = 0.0
                    lmpars['k4'].value = 0

                groups = MC_GROUPS
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, radial_time, scaled_stress, g=self.g, h=self.h)

                if len(mod_spike_time) == 0:
                    iff_data[distance_from_center] = None
                    continue

                if len(mod_spike_time) != len(mod_fr_inst):
                    if len(mod_fr_inst) > 1:
                        mod_fr_inst_interp = np.interp(mod_spike_time, radial_time, mod_fr_inst)
                    else:
                        mod_fr_inst_interp = np.zeros_like(mod_spike_time)
                else:
                    mod_fr_inst_interp = mod_fr_inst

                features, _ = pop_model(mod_spike_time, mod_fr_inst_interp)

                iff_data[distance_from_center] = {
                    'afferent_type': self.aff_type,
                    'num_of_spikes': len(mod_spike_time),
                    'mean_firing_frequency': features["Average Firing Rate"],
                    'peak_firing_frequency': np.max(mod_fr_inst_interp),
                    'first_spike_time': mod_spike_time[0] if len(mod_spike_time) > 0 else None,
                    'last_spike_time': mod_spike_time[-1] if len(mod_spike_time) > 0 else None,
                    'Time': stress_data[distance_from_center]["Time"].tolist(),
                    'Stress': stress_data[distance_from_center]["Stress"].tolist(),
                    'mod_spike_time': mod_spike_time.tolist(),
                    'entire_iff': mod_fr_inst_interp.tolist()
                }

            self.radial_stress_data = stress_data
            self.radial_iff_data = iff_data
            
        except FileNotFoundError:
            logging.error(f"Could not find file: {radial_stress_file}")
            self.radial_stress_data = None
            self.radial_iff_data = None
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            self.radial_stress_data = None
            self.radial_iff_data = None

    def plot_spatial_coords(self, plot=False):
        """
        Plots the firing frequencies on a grid, with circle size and opacity based on peak firing frequency.
        """
        colors = {'SA': '#31a354', 'RA': '#3182bd'}
        plt.figure(figsize=(12, 8))

        x_positions = self.results.get("x_position")
        y_positions = self.results.get("y_position")
        mean_iffs = self.results.get("mean_firing_frequency")
        peak_iffs = self.results.get("peak_firing_frequency")

        x_positions = [float(value) for value in x_positions]
        y_positions = [float(value) for value in y_positions]
        alphas = [float(value) / max(peak_iffs) if value != 0 else 0 for value in peak_iffs]

        # Plot circles for each coordinate
        for x_pos, y_pos, radius, alpha in zip(x_positions, y_positions, peak_iffs, alphas):
            plt.gca().add_patch(
                patches.Circle((x_pos, y_pos), radius*2, edgecolor='black', 
                             facecolor=colors.get(self.aff_type), linewidth=1, alpha=0.5)
            )

        if plot:
            plt.xlabel('Length (mm)')
            plt.ylabel('Width (mm)')
            plt.title(f"{self.density if self.density else ''} {self.vf_tip_size} VF {self.aff_type} firing at {self.time_of_firing} ms Stress Distribution")

        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.xlim(4, 12)
        plt.ylim(2, 7)
        plt.xticks(range(4, 13), fontsize=32)
        plt.yticks(range(2, 8), fontsize=32)
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.5))

        plt.savefig(f"vf_graphs/spatial_plots/{self.density if self.density else ''}{self.vf_tip_size}_{self.aff_type}_{self.time_of_firing}_constant_opacity.png")
        plt.show()

    def run_single_unit_model_combined_graph(self, stress_threshold=0, plot=True):
        """
        Runs the single unit model and creates a combined graph of stress and firing rate.
        
        Parameters:
        - stress_threshold (float): Minimum stress value to consider
        - plot (bool): Whether to display the plot
        """
        if self.radial_stress_data is None:
            logging.error("radial_stress_data is None. Please run radial_stress_vf_model first.")
            return

        # Initialize variables for plotting
        legend = False
        common_stress_max = 0
        common_iff_max = 0
        SA_radius = 0

        if plot:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.subplots_adjust(hspace=0.5, wspace=0.4)
            axes = axes.flatten()

        # Process each radial distance
        for idx, (distance, data) in enumerate(self.radial_stress_data.items()):
            if idx >= 10:  # Only plot up to 10 entries
                break

            time = data["Time"]
            stress_values = data["Stress"]

            if np.max(stress_values) < stress_threshold:
                stress_values = np.zeros_like(stress_values)

            # Compute spikes using model
            lmpars = lmpars_init_dict['t3f12v3final']
            if self.aff_type == "RA":
                lmpars['tau1'].value = 8
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1
                lmpars['k1'].value = 35
                lmpars['k2'].value = 0
                lmpars['k3'].value = 0.0
                lmpars['k4'].value = 0

            groups = MC_GROUPS
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress_values, g=self.g, h=self.h)

            if len(mod_spike_time) == 0:
                break

            if distance > SA_radius:
                SA_radius = distance

            if len(mod_spike_time) != len(mod_fr_inst):
                if len(mod_fr_inst) > 1:
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
                else:
                    mod_fr_inst_interp = np.zeros_like(mod_spike_time)
            else:
                mod_fr_inst_interp = mod_fr_inst

            if not legend:
                common_stress_max = np.max(stress_values) + 50
                common_iff_max = np.max(mod_fr_inst_interp * 1e3) + 50

            if plot:
                ax = axes[idx]
                ax2 = ax.twinx()

                # Plot IFF (Hz)
                ax.plot(mod_spike_time, mod_fr_inst_interp * 1e3, label="IFF (Hz)", 
                       marker='o', linestyle='none', color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                ax.set_ylim(0, 525)

                # Plot Stress (kPa)
                ax2.plot(time, stress_values, label="Stress (kPa)", color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylim(0, 400)

                # Title and labels
                ax.set_title(f'Distance {distance:.2f} mm')
                ax.set_xlabel('Time (ms)')
                ax.set_xlim(0, max(time))

                if not legend:
                    ax.set_ylabel('Firing Rate (Hz)', color='blue')
                    ax2.set_ylabel('Stress (kPa)', color='red')
                    legend = True

        if plot:
            # Hide unused subplots
            for ax in axes[len(self.radial_stress_data):]:
                ax.axis('off')

            # Save plot
            os.makedirs("figure4", exist_ok=True)
            plt.savefig(f"figure4/radial_plots_{self.vf_tip_size}_{self.density if self.density else ''}_tauRI_{lmpars['tau1'].value}_g_{self.g}_h_{self.h}_a_{lmpars['k1'].value}_b_{lmpars['k2'].value}_c_{lmpars['k3'].value}.png")

        self.SA_radius = SA_radius


    def calculate_receptive_field_size(self, stress_threshold=0):
        """
        Calculates the receptive field size by finding the maximum distance where there is a response.
        
        Parameters:
        - stress_threshold (float): Minimum stress value to consider
        
        Returns:
        - float: The receptive field size
        """
        if self.radial_stress_data is None:
            logging.error("radial_stress_data is None. Please run radial_stress_vf_model first.")
            return None

        radius = 0

        # Process each radial distance
        for distance, data in self.radial_stress_data.items():
            time = data["Time"]
            stress_values = data["Stress"]

            if np.max(stress_values) < stress_threshold:
                stress_values = np.zeros_like(stress_values)

            # Compute spikes using model
            lmpars = lmpars_init_dict['t3f12v3final']
            if self.aff_type == "RA":
                lmpars['tau1'].value = 8
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1
                lmpars['k1'].value = 35
                lmpars['k2'].value = 0
                lmpars['k3'].value = 0.0
                lmpars['k4'].value = 0

            groups = MC_GROUPS
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress_values, g=self.g, h=self.h)

            # If there is a response, update the radius
            if len(mod_spike_time) > 0:
                if distance > radius:
                    radius = distance

        receptive_field_size = radius**2 * np.pi
        return receptive_field_size

    def get_model_results(self):
        """Returns the spatial model results."""
        return self.results

    def get_radial_iff_data(self):
        """Returns the radial stress and firing frequency data."""
        return self.radial_iff_data

    def get_SA_radius(self):
        """Returns the SA afferent radius."""
        return self.SA_radius
