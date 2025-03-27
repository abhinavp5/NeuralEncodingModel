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

def run_single_unit_model():
    """Code for running only the Interpolated Stress Model"""
        # if afferent_type = "RA":
        # # Use the correct RA parameters
        #     lmpars['tau1'].value = 8
        #     lmpars['tau2'].value = 200
        #     lmpars['tau3'].value = 1
        #     lmpars['k1'].value = 80
        #     lmpars['k2'].value = 0
        #     lmpars['k3'].value = 0.0001
        #     lmpars['k4'].value = 0
    og_data = pd.read_csv("data/updated_dense_interpolated_stress_trace_RA.csv")
    og_stress = og_data[og_data.columns[1]].to_numpy()


    vf_tip_sizes = [3.61, 4.08, 4.17, 4.31, 4.56]
    type_of_ramp = ["out", "shallow", "steep"]

    vf_list_len = len(vf_tip_sizes)
    ramps_len = len(type_of_ramp)

    fig, axs = plt.subplots(vf_list_len,ramps_len,figsize = (10,8),sharex= True, sharey= True)
    
    # variable for only adding th legend to the first plot
    legend_added = False

    #loop through the different tip_sizes
    for vf_idx,vf in enumerate(vf_tip_sizes):

        #loop through the type of ramp (normal, shallow(x*2), steep(x/2))
        for ramp_idx,ramp in enumerate(type_of_ramp):

            # data = pd.read_csv(f"data/vf_unscaled/{vf}_{ramp}.csv")
            try:
                data = pd.read_csv(f"data/P2/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv")
            except FileNotFoundError:
                logging.warning("The File was not found!")
                exit();
            

            # Define parameters for a single afferent simulation
            afferent_type = "SA" 
            time = data['Time (ms)'].to_numpy()

            #0.1 is the Scaling Factor in here
            scaling_factor = .28
            
            if (ramp == "out"):
                stress = scaling_factor * data[data.columns[1]].values
                # print(stress)
            elif(ramp == "shallow"):
                stress = scaling_factor * data[data.columns[1]].values
            elif (ramp == "steep"):
                stress = scaling_factor * data[data.columns[1]].values
                # print(stress)
            lmpars = lmpars_init_dict['t3f12v3final']
            if afferent_type == "RA" :
                lmpars['tau1'].value = 2.5
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1
                lmpars['k1'].value = 35
                lmpars['k2'].value = 0
                lmpars['k3'].value = 0.0
                lmpars['k4'].value = 0
            elif afferent_type == "SA":
                lmpars['tau1'].value = 2.5
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1744.6
                lmpars['tau4'].value = np.inf
                lmpars['k1'].value = .74
                lmpars['k2'].value = 2.75
                lmpars['k3'].value = .07
                lmpars['k4'].value = .0312

            #'RA': {'tau1': 2.5, 'tau2': 200, 'tau3': 1, 'k1': 35, 'k2': 0, 'k3': 0, 'k4': 0}}
            groups = MC_GROUPS
            if afferent_type == "SA":
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= 0.2, h = .5)
            elif afferent_type == "RA":
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= 0.4, h = 0.1)
            # check for if the spikes are generated
            if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
                logging.warning(f"SPIKES COULDNT NOT BE GENRERATED on {vf} and {ramp} ")
                return
            # checking if the lenghts are equal for plotting
            if len(mod_spike_time) != len(mod_fr_inst):
                if len(mod_fr_inst) > 1:
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
                else:
                    print("note enough data to mod_fr_interp")
                    mod_fr_inst_interp = np.zeros_like(mod_spike_time)
            else:
                mod_fr_inst_interp = mod_fr_inst

            # print(f"INTERPOLATED DATA:", mod_fr_inst_interp* 1e3)

            
            print("MAX TIME:", np.max(time))

            # Plotting Firing Rate & stress
            axs[vf_idx, ramp_idx].plot(mod_spike_time, mod_fr_inst_interp * 1e3, label="Firing Rate (Hz)", marker='o', linestyle='none')
            axs[vf_idx, ramp_idx].plot(time, stress, label="Stress (kPa)", color="red")
            axs[vf_idx, ramp_idx].set_title(f"{vf} {ramp} {afferent_type} Afferent")
            axs[vf_idx, ramp_idx].set_ylabel('Firing Rate (Hz) / Stress')



            if legend_added is False:
                axs[vf_idx, ramp_idx].legend()
                legend_added = True

    plt.tight_layout()
    plt.show()

'''
shallow, LIF_RESOLUTION == 2
out, LIF_RESOLUTION == 1
steep, LIF_RESOLUTION == .5
'''

def run_same_plot(afferent_type, ramp, scaling_factor = .1, plot_style = "smooth", smoothing_type = "gaussian"):
    """
    Run the single unit model and plot results.
    
    Parameters:
    -----------
    afferent_type : str
        Type of afferent ("SA" or "RA")
    ramp : str
        Type of ramp ("shallow", "out", or "steep")
    scaling_factor : float
        Scaling factor for stress values
    plot_style : str
        Plot style ("smooth" or "points")
    smoothing_type : str
        Type of smoothing to use ("gaussian" or "bezier")
    """
    def moving_average(data, window_size= 1):
        return np.convolve(data, np.ones(window_size)/window_size, mode = 'valid')

    colors = ['#440154', '#3b528b', '#21908c', '#5dc963', '#fde725']
    vf_tip_sizes = [3.61, 4.08, 4.17, 4.31, 4.56]  # The five tip sizes to plot
    vf_list_len = len(vf_tip_sizes)
    LifConstants.set_resolution(1)

    fig, axs = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    for vf, color in zip(vf_tip_sizes,colors):
        try: 
            #This is the previously used Unscaled Data
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

        # After processing but before plotting, print the IFFs
        print(f"\nVF tip size: {vf}")
        print(f"Number of spikes: {len(mod_spike_time)}")
        print(f"Time points (ms): {mod_spike_time[:10]}... (showing first 10)")
        print(f"IFFs (Hz): {mod_fr_inst_interp[:10] * 1e3}... (showing first 10)")
        print(f"Max IFF: {np.max(mod_fr_inst_interp) * 1e3:.2f} Hz")
        print(f"Mean IFF: {np.mean(mod_fr_inst_interp) * 1e3:.2f} Hz")

        #Rough Case (Unsmoothed Lines)
        axs[0].plot(time, stress, label = f"{vf}", color = color)

        if plot_style == "points":
            # Plot individual points without smoothing
            axs[1].plot(mod_spike_time, mod_fr_inst_interp * 1e3, color=color, label=f"{vf}", 
                      marker="o", linestyle="none")
        else:  # plot_style == "smooth"
            if smoothing_type == "bezier":
                # Create points for Bezier curve
                points = np.column_stack((mod_spike_time, mod_fr_inst_interp * 1e3))
                # Calculate Bezier curve
                x_curve, y_curve = bezier_curve(points)
                # Plot the Bezier curve
                axs[1].plot(x_curve, y_curve, color=color, label=f"{vf}", 
                          marker="", linestyle="solid")
            else:  # smoothing_type == "gaussian"
                # Create a smoothing mask - regions where we WANT to apply smoothing
                # We want to avoid smoothing in 0-50ms, 1000-4000ms, and 4950ms+
                if vf == 3.61 and afferent_type == "SA":
                    smoothing_regions = (
                        ((mod_spike_time >= 457) & (mod_spike_time < 847)) | 
                        ((mod_spike_time >= 4582) & (mod_spike_time < 4862))
                    )
                elif vf == 4.08 and afferent_type == "SA":
                    smoothing_regions = (
                        ((mod_spike_time >= 50) & (mod_spike_time < 1000)) | 
                        ((mod_spike_time >= 4000) & (mod_spike_time < 4950))
                    )
                elif vf == 4.17 and afferent_type == "SA":
                    smoothing_regions = (
                        ((mod_spike_time >= 12.5) & (mod_spike_time < 1000)) | 
                        ((mod_spike_time >= 4000) & (mod_spike_time < 4950))
                    )
                elif vf == 4.31 and afferent_type == "SA":
                    smoothing_regions = (
                        ((mod_spike_time >= 12.5) & (mod_spike_time < 1000)) | 
                        ((mod_spike_time >= 4000) & (mod_spike_time < 4950))
                    )
                elif vf == 4.56 and afferent_type == "SA"   :
                    smoothing_regions = (
                        ((mod_spike_time >= 12.5) & (mod_spike_time < 1000)) | 
                        ((mod_spike_time >= 4000) & (mod_spike_time < 4950))
                    )
                elif vf ==3.61 and afferent_type == "RA":
                    smoothing_regions = (
                        ((mod_spike_time >= 12.5) & (mod_spike_time < 1000)) | 
                        ((mod_spike_time >= 4000) & (mod_spike_time < 4950))
                    )
                elif vf == 4.08 and afferent_type == "RA":
                    smoothing_regions = (
                        ((mod_spike_time >= 12.5) & (mod_spike_time < 1000)) | 
                        ((mod_spike_time >= 4000) & (mod_spike_time < 4950))
                    )
                elif vf == 4.17 and afferent_type == "RA":
                    smoothing_regions = (
                        ((mod_spike_time >= 12.5) & (mod_spike_time < 1000)) | 
                        ((mod_spike_time >= 4000) & (mod_spike_time < 4950))
                    )   
                elif vf == 4.31 and afferent_type == "RA":
                    smoothing_regions = (
                        ((mod_spike_time >= 12.5) & (mod_spike_time < 1000)) | 
                        ((mod_spike_time >= 4000) & (mod_spike_time < 4950))
                    )
                elif vf == 4.56 and afferent_type == "RA":
                    smoothing_regions = (
                        ((mod_spike_time >= 12.5) & (mod_spike_time < 1000)) | 
                        ((mod_spike_time >= 4000) & (mod_spike_time < 4950))
                    )
                    
                    


            # Copy original firing rate data
            final_fr = mod_fr_inst_interp.copy()
            
            # Apply Gaussian smoothing only in smoothing regions
            if np.any(smoothing_regions):
                # Apply smoothing only to selected time ranges
                smooth_regions = gaussian_filter1d(mod_fr_inst_interp[smoothing_regions], sigma=7)
                final_fr[smoothing_regions] = smooth_regions
            
            # Plot the mixed smoothed and unsmoothed data
            axs[1].plot(mod_spike_time, final_fr * 1e3, color=color,
                        label=f"{vf}", marker="", linestyle="solid")
            


    # Configure both axes
    for ax in axs:
        ax.set_xlim(left=0)  # Set x-axis to start at 0
        ax.set_ylim(bottom=0)  # Set y-axis to start at 0
        ax.minorticks_on()  # Enable minor ticks
        ax.grid(True, which='major', linestyle='-', alpha=0.3)  # Add major grid lines
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)  # Add minor grid lines

    axs[0].set_title(f"{afferent_type} Von Frey Stress Traces")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Stress (kPa)")
    axs[0].legend(loc = "best")

    axs[1].set_title(f"{afferent_type} IFF's associated with Stress Traces")
    axs[1].set_xlabel("Spike Time (ms)")
    axs[1].set_ylabel("Firing Rate (kHz)")
    axs[1].legend(loc = "best")
    
    # Print out all time stamps and firing frequencies
    print(f"\n==== Time stamps and firing frequencies for {afferent_type} afferent ====")
    for vf in vf_tip_sizes:
        try:
            # Load the data again to match the same processing
            if vf == 4.56:
                data = pd.read_csv(f"data/P3/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv")
            else:
                data = pd.read_csv(f"data/P4/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv")
            
            time = data['Time (ms)'].to_numpy()
            stress = scaling_factor * data[data.columns[1]].values
            
            # Get the parameter values
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
            
            # Get the spike times and firing rates
            groups = MC_GROUPS
            if afferent_type == "SA":
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= 0.2, h = .5)
            elif afferent_type == "RA":
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= 0.4, h = 1)
            
            if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
                print(f"VF {vf}: No spikes generated")
                continue
            
            # Make sure lengths match
            if len(mod_spike_time) != len(mod_fr_inst):
                if len(mod_fr_inst) > 1:
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
                else:
                    mod_fr_inst_interp = np.zeros_like(mod_spike_time)
            else:
                mod_fr_inst_interp = mod_fr_inst
            
            # Save the time stamps and firing frequencies to a text file instead of printing
            output_filename = f"firing_data_{afferent_type}_{vf}_{ramp}_{scaling_factor}.txt"
            with open(output_filename, 'w') as f:
                f.write(f"VF {vf} - {len(mod_spike_time)} spikes:\n")
                f.write("Time (ms) | Firing Rate (Hz)\n")
                f.write("-" * 30 + "\n")
                for t, fr in zip(mod_spike_time, mod_fr_inst_interp * 1e3):
                    f.write(f"{t:.2f} | {fr:.2f}\n")
            print(f"Saved firing data for VF {vf} to {output_filename}")
        except Exception as e:
            print(f"Error processing VF {vf}: {str(e)}")
    
    plt.tight_layout()

    
    plt.savefig(f"vf_graphs/stress_iffs_different_plot/{afferent_type}_{ramp}_{scaling_factor}_{plot_style}_{smoothing_type}.png")
    plt.savefig(f"Figure1/{afferent_type}_{ramp}_{scaling_factor}_{plot_style}_{smoothing_type}.png")
    plt.show()

"""
Code for running only the Interpolated Stress Model and plotting 
IFF and stress on the same graph for a single ramp type. INitially had a 5x3 chart
but I couldnt reliably change the the constant LIF_RESOLUTION for every different
type of ramp & resolution:
shallow, LIF_RESOLUTION == 2
out, LIF_RESOLUTION == 1
steep, LIF_RESOLUTION == .5

Inputs:
afferent_type: "RA" or "SA"
ramp: "shallow", "out", or "steep"
"""
def run_single_unit_model_combined_graph(afferent_type, ramp):

    
    vf_tip_sizes = [3.61, 4.08, 4.17, 4.31, 4.56]  # The five tip sizes to plot
    vf_list_len = len(vf_tip_sizes)

    # Create subplots for firing rate and stress in a 5x1 layout
    fig, axs = plt.subplots(vf_list_len, 1, figsize=(8, 10), sharex=True, sharey=True)

    legend_added = False
    for vf_idx, vf in enumerate(vf_tip_sizes):
        data = pd.read_csv(f"data/vf_unscaled/{vf}_{ramp}.csv")

        time = data['Time (ms)'].to_numpy()

        # Scaling factor
        scaling_factor = 0.28
        print("MAX STRESS", np.max(data[data.columns[1]].values))
        stress = scaling_factor * data[data.columns[1]].values

        lmpars = lmpars_init_dict['t3f12v3final']
        if afferent_type == "RA":
            lmpars['tau1'].value = 8
            lmpars['tau2'].value = 200
            lmpars['tau3'].value = 1
            lmpars['k1'].value = 35
            lmpars['k2'].value = 0
            lmpars['k3'].value = 0.0
            lmpars['k4'].value = 0

        groups = MC_GROUPS
        if afferent_type == "SA":
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= 0.2, h = .5)
        elif afferent_type == "RA":
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= 0.2, h = 5)

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

        axs[vf_idx].plot(mod_spike_time, mod_fr_inst_interp * 1e3, label="Firing Rate (Hz)", marker='o', linestyle='none')
        axs[vf_idx].plot(time, stress, label="Stress (kPa)", color="red")
        axs[vf_idx].set_title(f"{vf} {ramp} {afferent_type} Afferent")
        axs[vf_idx].set_ylabel('Firing Rate (Hz) / Stress (kPa)')
        
        if not legend_added:
            axs[vf_idx].legend()
            legend_added = True

    fig.suptitle(f"Firing Rate and Stress for Ramp Type: {ramp}")
    plt.tight_layout()
def sa_shallow_steep_stacked(vf_tip_size, afferent_type, scaling_factor):
        types_of_ramp = ["shallow", "steep"]
        fig, axs = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    
        for ramp_idx, ramp in enumerate(types_of_ramp):
            if ramp == ("shallow"):
                LifConstants.set_resolution(2)

            elif ramp == ("steep"):
                LifConstants.set_resolution(.5)

            
            data = pd.read_csv(f"data/vf_unscaled/{vf_tip_size}_{ramp}.csv")
            time = data['Time (ms)'].to_numpy()
            stress = scaling_factor * data[data.columns[1]].values

            lmpars = lmpars_init_dict['t3f12v3final']
            if afferent_type == "RA":
                lmpars['tau1'].value = 8
                lmpars['tau2'].value = 200
                lmpars['tau3'].value = 1
                lmpars['k1'].value = 35
                lmpars['k2'].value = 0
                lmpars['k3'].value = 0.0
                lmpars['k4'].value = 0

            groups = MC_GROUPS
            if afferent_type == "SA":
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= 0.2, h = .5)
            elif afferent_type == "RA":
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g= .2, h = 0.5)

            if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
                logging.warning(f"SPIKES COULD NOT BE GENERATED on {vf_tip_size} and {ramp}")
                continue
            if len(mod_spike_time) != len(mod_fr_inst):
                if len(mod_fr_inst) > 1:
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
                else:
                    mod_fr_inst_interp = np.zeros_like(mod_spike_time)
            else:
                mod_fr_inst_interp = mod_fr_inst
            
            axs[0].plot(time, stress, label = f"{ramp}")

            axs[1].plot(mod_spike_time, mod_fr_inst_interp * 1e3, label = f"{ramp}", marker = "o", linestyle = "none")
            
        axs[0].set_title(f"{vf_tip_size}mm {afferent_type} Von Frey Stress Traces with scaling factor = {scaling_factor}")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Stress (kPa)")
        axs[0].legend(loc = "best")

        axs[1].set_title(f"{afferent_type} Steep and Shallow IFF's associated with Stress Traces")
        axs[1].set_xlabel("Spike Time (ms)")
        axs[1].set_ylabel("Firing Rate (kHz)")
        axs[1].legend(loc = "best")
        axs[1].set_xlim([0, 5000])
        

        plt.tight_layout()
        # plt.show()

        plt.savefig(f"shallow_steep_same_plot/{vf_tip_size}mm_{afferent_type}_stacked_{scaling_factor}.png")

def bezier_curve(points, num_points=100):
    """
    Calculate a Bezier curve through a set of points.
    
    Parameters:
    -----------
    points : array-like
        Array of points (x, y) to fit the curve through
    num_points : int
        Number of points to generate on the curve
    
    Returns:
    --------
    tuple
        (x_coords, y_coords) of the Bezier curve
    """
    points = np.array(points)
    n = len(points) - 1
    
    # Generate parameter values
    t = np.linspace(0, 1, num_points)
    
    # Initialize arrays for x and y coordinates
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    
    # Calculate Bezier curve using Bernstein polynomials
    for i in range(n + 1):
        # Binomial coefficient
        binom = np.math.comb(n, i)
        
        # Bernstein polynomial
        bern = binom * (t ** i) * ((1 - t) ** (n - i))
        
        # Add contribution from this point
        x += points[i, 0] * bern
        y += points[i, 1] * bern
    
    return x, y

def main():
    parser = argparse.ArgumentParser(description='Plot single unit model results.')
    parser.add_argument('afferent_type', choices=['SA', 'RA'], help='Type of afferent')
    parser.add_argument('ramp', choices=['shallow', 'out', 'steep'], help='Type of ramp')
    parser.add_argument('--scaling_factor', type=float, default=0.1, help='Scaling factor for stress values')
    parser.add_argument('--plot_style', choices=['smooth', 'points'], default='smooth', help='Plot style')
    parser.add_argument('--smoothing_type', choices=['gaussian', 'bezier'], default='gaussian', help='Type of smoothing to use')
    
    args = parser.parse_args()
    
    run_same_plot(
        args.afferent_type,
        args.ramp,
        scaling_factor=args.scaling_factor,
        plot_style=args.plot_style,
        smoothing_type=args.smoothing_type
    )

if __name__ == '__main__':
    main()
