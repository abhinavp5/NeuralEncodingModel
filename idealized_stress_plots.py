'''
Script for plotting idealized stress traces and their corresponding firing rates
using the same logic as single_unit_plots.py'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from lmfit import minimize, fit_report, Parameters
from aim2_population_model_spatial_aff_parallel import get_mod_spike
from model_constants import (MC_GROUPS, LifConstants)
import argparse
import os

#Global Variables
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

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20  # Very big font

def plot_idealized_stress(afferent_type):
    # Create directory for CSV files if it doesn't exist
    csv_dir = "spike_data"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # Define colors for different Von Frey sizes
    colors = ['#440154', '#3b528b', '#21908c']  # Colors for 3.61, 4.17, 4.56
    vf_tip_sizes = [3.61, 4.17, 4.56]
    
    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    
    for vf, color in zip(vf_tip_sizes, colors):
        try:
            # Read idealized stress data
            data = pd.read_csv(f"idealized_stress/{vf}_ideal_stress.csv")
            logging.warning(f"Reading data for {vf}")
            time = np.ceil(data['Time (ms)'].to_numpy() * 1000)
            stress =  data['Stress (kPa)'].values  # Scale stress by 2x
            logging.warning(f"Max stress: {np.max(stress)}")

        except KeyError as e:
            logging.warning(f"File not found for {vf}")
            continue

        # Set parameters based on afferent type
        lmpars = lmpars_init_dict['t3f12v3final']
        if afferent_type == "RA":
            lmpars['tau1'].value = 20  # Changed from 2.5
            lmpars['tau2'].value = 200
            lmpars['tau3'].value = 1
            lmpars['k1'].value = 35
            lmpars['k2'].value = 0
            lmpars['k3'].value = 0.0
            lmpars['k4'].value = 0
        elif afferent_type == "SA":
            lmpars['tau1'].value = 8
            lmpars['tau2'].value = 200
            lmpars['tau3'].value = 1 
            lmpars['tau4'].value = np.inf
            lmpars['k1'].value = 0.74
            lmpars['k2'].value = 1.0
            lmpars['k3'].value = 0.07
            lmpars['k4'].value = 0.0312

        # Get firing rates
        groups = MC_GROUPS
        if afferent_type == "SA":
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=0.2, h=0.5)
        elif afferent_type == "RA":
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=0.2, h=.5)
            mod_fr_inst = mod_fr_inst * 2

        logging.warning(f"VF:{vf} {len(mod_spike_time)}")
        if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
            logging.error(f"SPIKES COULD NOT BE GENERATED for {vf}")
            continue

        # Interpolate firing rates if needed
        if len(mod_spike_time) != len(mod_fr_inst):
            if len(mod_fr_inst) > 1:
                mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
            else:
                mod_fr_inst_interp = np.zeros_like(mod_spike_time)
        else:
            mod_fr_inst_interp = mod_fr_inst

        # Save spike times and firing rates to CSV
        spike_data = pd.DataFrame({
            'spike_time_ms': mod_spike_time,
            'firing_rate_kHz': mod_fr_inst_interp * 1e3
        })
        csv_filename = f"{csv_dir}/{afferent_type}_idealized_vf{vf}.csv"
        spike_data.to_csv(csv_filename, index=False)
        logging.warning(f"Saved spike data to {csv_filename}")

        # Plot stress traces
        axs[0].plot(time, stress, label=f"{vf}", color=color)

        # Plot firing rates
        axs[1].plot(mod_spike_time, mod_fr_inst_interp * 1e3, color=color, label=f"{vf}", 
                   marker="o", markersize=4, linestyle="none")

    # Configure axes
    for ax in axs:
        ax.set_xlim(left=0)
        ax.minorticks_on()

    # Set specific y-axis bounds for each subplot
    axs[0].set_ylim(bottom=0, top=400)  # For stress traces
    axs[1].set_ylim(bottom=0, top=400)  # For firing rates

    # Set labels and titles
    axs[0].set_title(f"{afferent_type} Idealized Stress Traces")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Stress (kPa)")
    axs[0].legend(loc="best")

    axs[1].set_title(f"{afferent_type} IFF's associated with Idealized Stress Traces")
    axs[1].set_xlabel("Spike Time (ms)")
    axs[1].set_ylabel("Firing Rate (Hz)")
    axs[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(f"idealized_stress/plots/{afferent_type}_idealized_scaling.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot idealized stress traces and firing rates.')
    parser.add_argument('afferent_type', choices=['SA', 'RA'], help='Type of afferent (SA or RA)')
    
    args = parser.parse_args()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists("idealized_stress/plots"):
        os.makedirs("idealized_stress/plots")
    
    plot_idealized_stress(
        args.afferent_type,
    )

if __name__ == '__main__':
    main()
