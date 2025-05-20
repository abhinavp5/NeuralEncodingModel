"""
Raw Data Plotting Module for Afferent Response Analysis

This module generates plots of firing rates calculated from stress traces for different von Frey filament sizes.
The plotting methodology follows these steps:

1. Data Preparation:
   - Reads stress data from CSV files for each von Frey filament size (3.61, 4.08, 4.31, 4.56 mm)
   - Each file contains Time (ms) and three stress measurements: Average, Upper bound, and Lower bound
   - Data is used directly from the CSV files without interpolation

2. Firing Rate Calculation:
   - Uses a Leaky Integrate-and-Fire (LIF) model to convert stress to firing rates
   - Parameters are set differently for SA (Slowly Adapting) and RA (Rapidly Adapting) afferents
   - For each von Frey size, three firing rate traces are calculated:
     * Average stress → Average firing rate
     * Upper bound stress → Upper firing rate
     * Lower bound stress → Lower firing rate

3. Plot Generation:
   - Creates a single figure with all von Frey sizes
   - For each size:
     * Plots three lines: upper bound (dashed), average (solid), and lower bound (dashed)
     * Shades the area between upper and lower bounds
     * Uses different colors for each von Frey size
   - Includes proper axis labels, limits, and legend
   - Saves the plot as a high-resolution PNG file

Usage:
    python raw_data_plots.py --afferent_type [SA|RA]

Output:
    Saves plots to 'raw_data_plots/raw_firing_frequencies_[SA|RA].png'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from lmfit import minimize, fit_report, Parameters
from aim2_population_model_spatial_aff_parallel import get_mod_spike
from model_constants import (MC_GROUPS, LifConstants)
import argparse
import os
import traceback
import sys

# Disable matplotlib debug logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Set up parameters for plotting
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20  # Very big font

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

logging.basicConfig(level=logging.WARNING)

def calculate_firing_rates(time, stress, lmpars, g, h):
    """Calculate firing rates for a given stress trace."""
    groups = MC_GROUPS
    mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=g, h=h)
    

    # Convert to Hz and ensure non-negative values
    rates = mod_fr_inst * 1e3
    rates = np.maximum(rates, 0)
    
    return mod_spike_time, rates

def plot_raw_data(afferent_type):
    """
    Plot raw firing frequency data calculated from stress traces for different von Frey filament sizes.
    Also plot the stress traces with fill between upper and lower bounds and the median as a solid line.
    Args:
        afferent_type (str): Type of afferent ('SA' or 'RA')
    """
    # Define the von Frey filament sizes
    vf_sizes = [3.61, 4.08, 4.31, 4.56]
    
    # Define colors for different von Frey sizes (in the same order as vf_sizes)
    colors = ['#3D2674', '#6677FA', '#FF9047', '#92CA68']
    
    # Define figure parameters to match single_unit_plots.py
    fig_params = {
        'figsize': (14, 6),
        'left': 0.098,
        'right': 0.95,
        'top': 0.937,
        'bottom': 0.097
    }
    
    # --- Firing Frequency Plot ---
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 1,
                         left=0.098,
                         right=0.95,
                         top=0.937,
                         bottom=0.097)
    ax = fig.add_subplot(gs[0])
    
    # --- Stress Plot ---
    fig_stress = plt.figure(figsize=fig_params['figsize'])
    gs_stress = fig_stress.add_gridspec(1, 1,
                         left=fig_params['left'],
                         right=fig_params['right'],
                         top=fig_params['top'],
                         bottom=fig_params['bottom'])
    ax_stress = fig_stress.add_subplot(gs_stress[0])
    
    # Set x and y limits for stress plot
    ax_stress.set_xlim(0, 5000)
    ax_stress.set_ylim(0, 400)
    
    for i, vf_size in enumerate(vf_sizes):
        try:
            print(f"\nProcessing von Frey size: {vf_size}")
            
            # Read the raw stress data
            data = pd.read_csv(f"aggregated_data/{vf_size}_raw_agg_stress.csv")
            print(f"Data shape: {data.shape}")
            print(f"Columns: {data.columns}")
            
            time = data["Time (ms)"].to_numpy()
            avg_stress = data["Avg stress (kPa)"].to_numpy()
            upper_stress = data["Upper (kPa)"].to_numpy()
            lower_stress = data["Lower (kPa)"].to_numpy()
            
            # --- Stress Plot ---
            ax_stress.fill_between(time, lower_stress, upper_stress, color=colors[i], alpha=0.2)
            ax_stress.plot(time, avg_stress, color=colors[i], linewidth=2, label=f'{vf_size}')

            lmpars = lmpars_init_dict['t3f12v3final']
            # Set parameters based on afferent type
            if afferent_type == "RA":
                lmpars['tau1'].value = 2.5
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1
                lmpars['k1'].value = 35
                lmpars['k2'].value = 0
                lmpars['k3'].value = 0.0
                lmpars['k4'].value = 0
                g, h = 0.4, 1.0
            else:  # SA
                lmpars['tau1'].value = 8
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1744.6
                lmpars['tau4'].value = np.inf
                lmpars['k1'].value = 0.74
                lmpars['k2'].value = 1.0  # Updated from 0.2088
                lmpars['k3'].value = 0.07
                lmpars['k4'].value = 0.0312
                g, h = 0.2, 0.5
            
            # Calculate firing rates for all three stress traces
            avg_time, avg_rates = calculate_firing_rates(time, avg_stress, lmpars, g, h)
            upper_time, upper_rates = calculate_firing_rates(time, upper_stress, lmpars, g, h)
            lower_time, lower_rates = calculate_firing_rates(time, lower_stress, lmpars, g, h)
            
            # Create smooth interpolated curves
            smooth_time = np.linspace(0, 5000, 5000)
            avg_rates_smooth = np.interp(smooth_time, avg_time, avg_rates)
            upper_rates_smooth = np.interp(smooth_time, upper_time, upper_rates)
            lower_rates_smooth = np.interp(smooth_time, lower_time, lower_rates)
            
            # --- Firing Frequency Plot ---
            ax.fill_between(smooth_time, lower_rates_smooth, upper_rates_smooth, color=colors[i], alpha=0.2)
            ax.plot(smooth_time, upper_rates_smooth, color=colors[i], linewidth=1, alpha=0.5)
            ax.plot(smooth_time, avg_rates_smooth, color=colors[i], linewidth=1.5, label=f'{vf_size}mm')
            ax.plot(smooth_time, lower_rates_smooth, color=colors[i], linewidth=1, alpha=0.5)
            
            
        except Exception as e:
            print(f"Error processing von Frey size {vf_size}: {str(e)}")
            print(f"Full error: {traceback.format_exc()}")
            continue
    
    # --- Firing Frequency Plot Formatting ---
    ax.set_xlim(0,5000)
    ax.set_ylim(0,275)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    os.makedirs('raw_data_plots', exist_ok=True)
    plt.savefig(f'raw_data_plots/raw_firing_frequencies_{afferent_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- Stress Plot Formatting ---
    ax_stress.set_xlim(0,5000)
    ax_stress.set_ylim(0,400)
    ax_stress.set_xlabel('Time (ms)', fontsize=14)
    ax_stress.set_ylabel('Stress (kPa)', fontsize=14)
    ax_stress.spines['top'].set_visible(True)
    ax_stress.spines['right'].set_visible(True)
    plt.savefig(f'raw_data_plots/raw_stress_traces_{afferent_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

'''
    # --- Median Firing Rate Plot for SA ---
    fig_SA = plt.figure(figsize=fig_params['figsize'])
    gs_SA = fig_SA.add_gridspec(1, 1,
                         left=fig_params['left'],
                         right=fig_params['right'],
                         top=fig_params['top'],
                         bottom=fig_params['bottom'])
    ax_SA = fig_SA.add_subplot(gs_SA[0])
    for i, vf_size in enumerate(vf_sizes):
        try:
            data = pd.read_csv(f"aggregated_data/{vf_size}_raw_agg_stress.csv")
            time = data["Time (ms)"].to_numpy()
            avg_stress = data["Avg stress (kPa)"].to_numpy()
            lmpars = lmpars_init_dict['t3f12v3final']
            # SA parameters
            lmpars['tau1'].value = 8
            lmpars['tau2'].value = 200
            lmpars['tau3'].value = 1744.6
            lmpars['tau4'].value = np.inf
            lmpars['k1'].value = 0.74
            lmpars['k2'].value = 1.0
            lmpars['k3'].value = 0.07
            lmpars['k4'].value = 0.0312
            g, h = 0.2, 0.5
            median_time, median_rates = calculate_firing_rates(time, avg_stress, lmpars, g, h)
            smooth_time = np.linspace(0, 5000, 5000)
            median_rates_smooth = np.interp(smooth_time, median_time, median_rates)
            ax_SA.plot(smooth_time, median_rates_smooth, color=colors[i], marker='o', linestyle='none', markersize=4, label=f'{vf_size}')
        except Exception as e:
            print(f"Error processing SA median for von Frey size {vf_size}: {str(e)}")
            continue
    ax_SA.set_xlim(0, 5000)
    ax_SA.set_ylim(0, 225)
    ax_SA.set_xlabel('Time (ms)', fontsize=14)
    ax_SA.set_ylabel('Firing Rate (Hz)', fontsize=14)
    ax_SA.legend(loc='best', frameon=False)
    ax_SA.spines['top'].set_visible(True)
    ax_SA.spines['right'].set_visible(True)
    plt.savefig('raw_data_plots/median_firing_SA.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- Median Firing Rate Plot for RA ---
    fig_RA = plt.figure(figsize=fig_params['figsize'])
    gs_RA = fig_RA.add_gridspec(1, 1,
                         left=fig_params['left'],
                         right=fig_params['right'],
                         top=fig_params['top'],
                         bottom=fig_params['bottom'])
    ax_RA = fig_RA.add_subplot(gs_RA[0])
    for i, vf_size in enumerate(vf_sizes):
        try:
            data = pd.read_csv(f"aggregated_data/{vf_size}_raw_agg_stress.csv")
            time = data["Time (ms)"].to_numpy()
            avg_stress = data["Avg stress (kPa)"].to_numpy()
            lmpars = lmpars_init_dict['t3f12v3final']
            # RA parameters
            lmpars['tau1'].value = 2.5
            lmpars['tau2'].value = 200
            lmpars['tau3'].value = 1
            lmpars['k1'].value = 35
            lmpars['k2'].value = 0
            lmpars['k3'].value = 0.0
            lmpars['k4'].value = 0
            g, h = 0.4, 1.0
            median_time, median_rates = calculate_firing_rates(time, avg_stress, lmpars, g, h)
            smooth_time = np.linspace(0, 5000, 5000)
            median_rates_smooth = np.interp(smooth_time, median_time, median_rates)
            ax_RA.plot(smooth_time, median_rates_smooth, color=colors[i], marker='o', linestyle='none', markersize=4, label=f'{vf_size}')
        except Exception as e:
            print(f"Error processing RA median for von Frey size {vf_size}: {str(e)}")
            continue
    ax_RA.set_xlim(0, 5000)
    ax_RA.set_ylim(0, 300)
    ax_RA.set_xlabel('Time (ms)', fontsize=14)
    ax_RA.set_ylabel('Firing Rate (Hz)', fontsize=14)
    ax_RA.legend(loc='best', frameon=False)
    ax_RA.spines['top'].set_visible(True)
    ax_RA.spines['right'].set_visible(True)
    plt.savefig('raw_data_plots/median_firing_RA.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
'''
def main():
    parser = argparse.ArgumentParser(description='Plot raw firing frequency data calculated from stress traces')
    parser.add_argument('--afferent_type', type=str, required=True, 
                      choices=['SA', 'RA'], help='Type of afferent (SA or RA)')
    args = parser.parse_args()
    
    plot_raw_data(args.afferent_type)

if __name__ == "__main__":
    main()
