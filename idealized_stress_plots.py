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
import traceback
import sys

# Disable all logging from imported modules
logging.getLogger('aim2_population_model_spatial_aff_parallel').setLevel(logging.ERROR)
logging.getLogger('stress_to_spike').setLevel(logging.ERROR)
logging.getLogger('gen_function').setLevel(logging.ERROR)
logging.getLogger('popul_model').setLevel(logging.ERROR)
logging.getLogger('fit_model_alt').setLevel(logging.ERROR)

# Configure logging to only show errors
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)

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

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20  # Very big font

def plot_idealized_stress():
    try:
        logging.info("Starting plot_idealized_stress function")
        # Create directory for CSV files if it doesn't exist
        csv_dir = "spike_data"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
            logging.info(f"Created directory: {csv_dir}")

        # Define colors for different Von Frey sizes
        colors = ['#4055C8', '#6BA7E0', '#8ACD76']  # Colors for 3.61, 4.17, 4.56
        vf_tip_sizes = [3.61, 4.17, 4.56]
        
        logging.info("Creating figures")
        # Create three separate figures, each matching the parameter plot dimensions
        figs = []
        axs = []
        
        # Create three separate figures
        for i in range(3):
            fig = plt.figure(figsize=(12, 5))  # Same size as parameter plot
            gs = fig.add_gridspec(1, 1,
                                left=0.098,    # left margin
                                right=0.95,    # right margin
                                top=0.937,     # top margin
                                bottom=0.097)  # bottom margin
            ax = fig.add_subplot(gs[0])
            figs.append(fig)
            axs.append(ax)
        
        # Dictionary to store IFFs for both SA and RA
        iffs = {'SA': {}, 'RA': {}}
        
        # First, collect all stress traces and generate IFFs
        stress_data = {}
        for vf, color in zip(vf_tip_sizes, colors):
            try:
                logging.info(f"Processing Von Frey size: {vf}")
                # Read idealized stress data
                data = pd.read_csv(f"idealized_stress/{vf}_ideal_stress.csv")
                logging.info(f"Successfully read data for {vf}")
                time = np.ceil(data['Time (ms)'].to_numpy() * 1000)
                stress = data['Stress (kPa)'].values
                stress_data[vf] = (time, stress)
                logging.info(f"Max stress for {vf}: {np.max(stress)}")

                # Generate IFFs for both SA and RA
                for aff_type in ['SA', 'RA']:
                    logging.info(f"Generating IFFs for {aff_type} afferent with Von Frey {vf}")
                    # Set parameters based on afferent type
                    lmpars = lmpars_init_dict['t3f12v3final']
                    if aff_type == "RA":
                        lmpars['tau1'].value = 20
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
                        lmpars['k2'].value = 1.0
                        lmpars['k3'].value = 0.07
                        lmpars['k4'].value = 0.0312
                        g, h = 0.2, 0.5

                    logging.info(f"Calling get_mod_spike for {aff_type} with parameters: g={g}, h={h}")
                    # Get firing rates
                    groups = MC_GROUPS
                    try:
                        mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=g, h=h)
                        logging.info(f"get_mod_spike completed for {aff_type}")
                    except Exception as e:
                        logging.error(f"Error in get_mod_spike: {str(e)}")
                        logging.error(traceback.format_exc())
                        continue

                    if len(mod_spike_time) > 0 and len(mod_fr_inst) > 0:
                        if len(mod_spike_time) != len(mod_fr_inst):
                            mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst) if len(mod_fr_inst) > 1 else np.zeros_like(mod_spike_time)
                        else:
                            mod_fr_inst_interp = mod_fr_inst
                        
                        iffs[aff_type][vf] = (mod_spike_time, mod_fr_inst_interp)

                        # Print firing rates and times
                        print(f"\nFiring rates for {aff_type} afferent with Von Frey {vf}:")
                        print("Time (ms)\tFiring Rate (Hz)")
                        print("-" * 30)
                        for t, fr in zip(mod_spike_time, mod_fr_inst_interp):
                            print(f"{t:.1f}\t\t{fr*1000:.2f}")

                        # Save spike data
                        spike_data = pd.DataFrame({
                            'spike_time_ms': mod_spike_time,
                            'firing_rate_kHz': mod_fr_inst_interp * 1e3
                        })
                        csv_filename = f"{csv_dir}/{aff_type}_idealized_vf{vf}.csv"
                        spike_data.to_csv(csv_filename, index=False)
                        logging.info(f"Saved spike data to {csv_filename}")

            except KeyError as e:
                logging.error(f"File not found for {vf}: {str(e)}")
                continue
            except Exception as e:
                logging.error(f"Error processing {vf}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        logging.info("Starting to plot data")
        # Plot stress traces (first figure)
        for vf, color in zip(vf_tip_sizes, colors):
            if vf in stress_data:
                time, stress = stress_data[vf]
                axs[0].plot(time, stress, label=f"{vf}", color=color)

        # Plot SA IFFs (second figure)
        for vf, color in zip(vf_tip_sizes, colors):
            if vf in iffs['SA']:
                mod_spike_time, mod_fr_inst_interp = iffs['SA'][vf]
                axs[1].plot(mod_spike_time, mod_fr_inst_interp * 1e3, color=color, label=f"{vf}",
                           marker="o", markersize=4, linestyle="none")

        # Plot RA IFFs (third figure)
        for vf, color in zip(vf_tip_sizes, colors):
            if vf in iffs['RA']:
                mod_spike_time, mod_fr_inst_interp = iffs['RA'][vf]
                axs[2].plot(mod_spike_time, mod_fr_inst_interp * 1e3, color=color, label=f"{vf}",
                           marker="o", markersize=4, linestyle="none")

        logging.info("Configuring plot axes")
        # Configure axes
        for i, ax in enumerate(axs):
            ax.set_xlim(left=0, right=5000)  # Set x-axis from 0-5000ms
            ax.minorticks_on()
            ax.set_xlabel("Time (ms)")

        # Set specific y-axis bounds for each subplot
        axs[0].set_ylim(bottom=0, top=400)  # For stress traces
        axs[1].set_ylim(bottom=0, top=250)  # For SA firing rates
        axs[2].set_ylim(bottom=0, top=400)  # For RA firing rates

        # Set titles and labels
        axs[0].set_title("Idealized Stress Traces")
        axs[1].set_title("SA IFF's")
        axs[2].set_title("RA IFF's")
        
        # Set y-axis labels
        axs[0].set_ylabel("Stress (kPa)")
        axs[1].set_ylabel("IFF (Hz)")
        axs[2].set_ylabel("IFF (Hz)")

        # Add legends
        for ax in axs:
            ax.legend(loc="best")

        logging.info("Saving plots")
        # Save each figure separately
        figs[0].savefig(f"idealized_stress/plots/stress_traces.png")
        figs[1].savefig(f"idealized_stress/plots/sa_iffs.png")
        figs[2].savefig(f"idealized_stress/plots/ra_iffs.png")
        
        logging.info("Showing plots")
        plt.show()
        logging.info("Plot function completed")
    except Exception as e:
        logging.error(f"Error in plot_idealized_stress: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description='Plot idealized stress traces and firing rates.')
        
        args = parser.parse_args()
        
        # Create plots directory if it doesn't exist
        if not os.path.exists("idealized_stress/plots"):
            os.makedirs("idealized_stress/plots")
        
        plot_idealized_stress()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
