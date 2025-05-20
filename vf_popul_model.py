import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import logging


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

            # Compute spikes using model (placeholder function `get_mod_spike`)
            mod_spike_time, mod_fr_inst = get_mod_spike(time, stress, g=self.g, h=self.h)

            if len(mod_spike_time) == 0:
                continue  # Skip if no spikes

            mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst) if len(mod_fr_inst) > 1 else np.zeros_like(mod_spike_time)
            peak_firing_freq = np.max(mod_fr_inst_interp)

            model_results["x_position"].append(row.iloc[0])
            model_results["y_position"].append(row.iloc[1])
            model_results["spike_timings"].append(mod_spike_time.tolist())
            model_results["mean_firing_frequency"].append(np.mean(mod_fr_inst_interp))
            model_results["peak_firing_frequency"].append(peak_firing_freq)
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
        if self.aff_type == "SA":
            self.g = 0.2
            self.h = 0.5
        elif self.aff_type == "RA":
            self.g = 0.4
            self.h = 1.0

        radial_stress_file = f"data/P2/{self.density}/{self.vf_tip_size}/{self.vf_tip_size}_radial_stress_corr_{self.density.lower()}.csv"
        logging.info(f"Attempting to read file: {radial_stress_file}")
        
        try:
            radial_stress = pd.read_csv(radial_stress_file)
            # Check if 'Time (ms)' column exists, if not try 'Time'
            time_col = 'Time (ms)' if 'Time (ms)' in radial_stress.columns else 'Time'
            radial_time = radial_stress[time_col].to_numpy()
            
            stress_data = {}
            iff_data = {}

            # Process each radial distance
            for col in radial_stress.columns:
                if col == time_col:  # Skip the time column
                    continue
                    
                matches = re.findall(r'\d\.\d{2}', col)
                if not matches:
                    continue  # Skip if distance format is incorrect

                distance_from_center = float(matches[0])
                scaled_stress = radial_stress[col] * self.sf
                stress_data[distance_from_center] = {
                    "Time": radial_time,
                    "Stress": scaled_stress.to_numpy()
                }

                # Compute spikes using model
                mod_spike_time, mod_fr_inst = get_mod_spike(radial_time, scaled_stress, g=self.g, h=self.h)

                if len(mod_spike_time) == 0:
                    iff_data[distance_from_center] = None
                    continue  # Skip if no spikes

                mod_fr_inst_interp = np.interp(mod_spike_time, radial_time, mod_fr_inst) if len(mod_fr_inst) > 1 else np.zeros_like(mod_spike_time)

                iff_data[distance_from_center] = {
                    'Time': stress_data[distance_from_center]["Time"].tolist(),
                    'Stress': stress_data[distance_from_center]["Stress"].tolist(),
                    'peak_firing_frequency': np.max(mod_fr_inst_interp),
                    'mod_spike_time': mod_spike_time.tolist(),
                    'entire_iff': mod_fr_inst_interp.tolist()
                }

            self.radial_stress_data = stress_data
            self.radial_iff_data = iff_data
            logging.info(f"Stored radial_stress_data with {len(stress_data)} entries")
            logging.info(f"Keys in radial_stress_data: {list(stress_data.keys())}")
            
        except FileNotFoundError:
            logging.error(f"Could not find file: {radial_stress_file}")
            self.radial_stress_data = None
            self.radial_iff_data = None
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            self.radial_stress_data = None
            self.radial_iff_data = None

    def get_model_results(self):
        """Returns the spatial model results."""
        return self.results

    def get_radial_iff_data(self):
        """Returns the radial stress and firing frequency data."""
        return self.radial_iff_data

    def get_SA_radius(self):
        """Returns the SA afferent radius."""
        return self.SA_radius

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

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Process each radial distance
        for distance, data in self.radial_stress_data.items():
            time = data["Time"]
            stress_values = data["Stress"]

            # Skip if stress is below threshold
            if np.max(stress_values) < stress_threshold:
                continue

            # Get firing data for this distance
            if distance in self.radial_iff_data and self.radial_iff_data[distance] is not None:
                firing_data = self.radial_iff_data[distance]
                mod_spike_time = np.array(firing_data["mod_spike_time"])
                mod_fr_inst = np.array(firing_data["entire_iff"])
            else:
                continue

            # Interpolate firing rate if needed
            if len(mod_spike_time) != len(mod_fr_inst):
                if len(mod_fr_inst) > 1:
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
                else:
                    mod_fr_inst_interp = np.zeros_like(mod_spike_time)
            else:
                mod_fr_inst_interp = mod_fr_inst

            # Update common limits for first iteration
            if not legend:
                common_stress_max = np.max(stress_values) + 50
                common_iff_max = np.max(mod_fr_inst_interp * 1e3) + 50

            # Plot data
            ax.plot(time, stress_values, '--', color='gray', alpha=0.5, label='Stress' if not legend else None)
            ax.plot(mod_spike_time, mod_fr_inst_interp * 1e3, 'o-', label=f'Distance {distance:.2f} mm')

        # Set plot properties
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing Rate (Hz) / Stress (kPa)')
        ax.set_title(f'Radial Stress and Firing Rate for {self.aff_type} Afferent')
        ax.legend()
        ax.grid(True)

        if plot:
            plt.show()

        return fig, ax
