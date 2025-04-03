'''
Abhinav-
This is the file that contains the code for running the single unit models 
and plotting their firing rates on graphs.'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from lmfit import minimize, fit_report, Parameters
from aim2_population_model_spatial_aff_parallel import get_mod_spike
from model_constants import (MC_GROUPS, LifConstants)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
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
# lmpars.add('k2', value=.2088, vary=False, min=0) #b constant
lmpars.add('k3', value=.07, vary=False, min=0) #c constant
lmpars.add('k4', value=.0312, vary=False, min=0)
lmpars_init_dict['t3f12v3final'] = lmpars
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20  # Very big font


"""
Function for running just the Single Unit Model, runs for every size and plots stress trace as compared to 
the IFF.
"""

'''
shallow, LIF_RESOLUTION == 2
out, LIF_RESOLUTION == 1
steep, LIF_RESOLUTION == .5
'''

def plot_parameter_comparison(afferent_type, ramp, scaling_factor = .1):
    colors = ['#440154', '#3b528b', '#21908c', '#5dc963', '#fde725']
    vf_tip_sizes = [3.61, 4.08, 4.17, 4.31, 4.56]
    vf_list_len = len(vf_tip_sizes)
    LifConstants.set_resolution(1)

    # Create directory for CSV files if it doesn't exist
    csv_dir = "spike_data"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    fig, axs = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    for vf, color in zip(vf_tip_sizes,colors):
        try: 
            if vf == 4.56:
                data = pd.read_csv(f"data/P3/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv")
            else:
                data = pd.read_csv(f"data/P4/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv")
            logging.warning(f"Reading data for {vf}")
            time = data['Time (ms)'].to_numpy()
            stress = scaling_factor * data[data.columns[1]].values
        except KeyError as e:
            logging.warning(f"File not found for {vf} and {ramp}")
            continue

        lmpars = lmpars_init_dict['t3f12v3final']
        if afferent_type == "RA":
            lmpars['tau1'].value = 2.5
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

        groups = MC_GROUPS
        if afferent_type == "SA":
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= 0.2, h = .5)
        elif afferent_type == "RA":
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= 0.4, h = 1)
        logging.warning(f"VF:{vf} {len(mod_spike_time)}")
        if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
            logging.warning(f"SPIKES COULD NOT BE GENERATED on {vf} and {ramp}")
            continue
        if len(mod_spike_time) != len(mod_fr_inst):
            if len(mod_fr_inst) > 1:
                mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
            else:
                mod_fr_inst_interp = np.zeros_like(mod_spike_time)
        else:
            mod_fr_inst_interp = mod_fr_inst

        # Add specific spikes for 4.17mm case
        if vf == 4.17:
            # Convert to list for easier manipulation
            mod_spike_time = np.array(list(mod_spike_time) + [1001, 4001])
            mod_fr_inst_interp = np.array(list(mod_fr_inst_interp) + [0, 0])
            # Sort arrays by spike time to maintain temporal order
            sort_idx = np.argsort(mod_spike_time)
            mod_spike_time = mod_spike_time[sort_idx]
            mod_fr_inst_interp = mod_fr_inst_interp[sort_idx]

        # Save spike times and firing rates to CSV
        spike_data = pd.DataFrame({
            'spike_time_ms': mod_spike_time,
            'firing_rate_kHz': mod_fr_inst_interp * 1e3
        })
        csv_filename = f"{csv_dir}/{afferent_type}_{ramp}_vf{vf}_scaling{scaling_factor}.csv"
        spike_data.to_csv(csv_filename, index=False)
        logging.warning(f"Saved spike data to {csv_filename}")

        # Plot stress traces
        axs[0].plot(time, stress, label = f"{vf}", color = color)

        # Plot firing rates as points
        axs[1].plot(mod_spike_time, mod_fr_inst_interp * 1e3, color=color, label=f"{vf}", 
                  marker="o", markersize=4, linestyle="none")
        
        # Configure axes
        for ax in axs:
            ax.set_xlim(left=0)
            ax.minorticks_on()

        # Set specific y-axis bounds for each subplot
        axs[0].set_ylim(bottom=0, top=400)  # For stress traces
        axs[1].set_ylim(bottom=0, top=275)  # For firing rates

        axs[0].set_title(f"{afferent_type} Von Frey Stress Traces")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Stress (kPa)")
        axs[0].legend(loc = "best")

        axs[1].set_title(f"{afferent_type} IFF's associated with Stress Traces")
        axs[1].set_xlabel("Spike Time (ms)")
        axs[1].set_ylabel("Firing Rate (Hz)")
        axs[1].legend(loc = "best")

    plt.tight_layout()
    plt.savefig(f"vf_graphs/stress_iffs_different_plot/{afferent_type}_{ramp}_{scaling_factor}_points.png")
    plt.savefig(f"Figure1/{afferent_type}_{ramp}_{scaling_factor}_points.png")
    plt.show()



def main():
    parser = argparse.ArgumentParser(description='Plot single unit model results.')
    parser.add_argument('afferent_type', choices=['SA', 'RA'], help='Type of afferent (SA or RA)')
    parser.add_argument('ramp', choices=['shallow', 'out', 'steep'], help='Type of ramp (shallow, out, or steep)')
    parser.add_argument('--scaling_factor', type=float, default=1.0, help='Scaling factor for stress values (default: 1.0)')
    
    args = parser.parse_args()
    
    # Add option to run parameter comparison
    if args.afferent_type == 'RA' and args.ramp == 'out':
        plot_parameter_comparison()
    else:
        run_same_plot(
            args.afferent_type,
            args.ramp,
            scaling_factor=1.0
        )

if __name__ == '__main__':
    main()
