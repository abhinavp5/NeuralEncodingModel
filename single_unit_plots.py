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

            # Add specific spikes for 4.17mm case
            if vf == 4.17:
                # Convert to list for easier manipulation
                mod_spike_time = np.array(list(mod_spike_time) + [1001, 3999])
                mod_fr_inst_interp = np.array(list(mod_fr_inst_interp) + [0, 0])
                # Sort arrays by spike time to maintain temporal order
                sort_idx = np.argsort(mod_spike_time)
                mod_spike_time = mod_spike_time[sort_idx]
                mod_fr_inst_interp = mod_fr_inst_interp[sort_idx]

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

def run_same_plot(afferent_type, ramp, scaling_factor = .1, plot_style = "smooth"):
    def moving_average(data, window_size= 1):
        return np.convolve(data, np.ones(window_size)/window_size, mode = 'valid')

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

        # Plot only points for IFFs without any smoothing
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

        # Add specific spikes for 4.17mm case
        if vf == 4.17:
            # Convert to list for easier manipulation
            mod_spike_time = np.array(list(mod_spike_time) + [1001, 3999])
            mod_fr_inst_interp = np.array(list(mod_fr_inst_interp) + [0, 0])
            # Sort arrays by spike time to maintain temporal order
            sort_idx = np.argsort(mod_spike_time)
            mod_spike_time = mod_spike_time[sort_idx]
            mod_fr_inst_interp = mod_fr_inst_interp[sort_idx]

        axs[vf_idx].plot(mod_spike_time, mod_fr_inst_interp * 1e3, label="Firing Rate (Hz)", marker='o', linestyle='none')
        axs[vf_idx].plot(time, stress, label="Stress (kPa)", color="red")
        axs[vf_idx].set_title(f"{vf} {ramp} {afferent_type} Afferent")
        axs[vf_idx].set_ylabel('Firing Rate (Hz) / Stress (kPa)')
        
        if not legend_added:
            axs[vf_idx].legend()
            legend_added = True

    fig.suptitle(f"Firing Rate and Stress for Ramp Type: {ramp}")
    plt.tight_layout()

def plot_parameter_comparison(afferent_type, ramp, param_name='tau1', param_values=None):
    # Colors from darkest to lightest gray, matching the image
    parameter_colors = ['black', '#666666', '#999999', '#CCCCCC']  # From darkest to lightest
    vf_tip_sizes = [3.61]
    vf_list_len = len(vf_tip_sizes)
    LifConstants.set_resolution(1)

    fig, axs = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    lmpars = lmpars_init_dict['t3f12v3final']
    
    # Use default tau1 values if no parameter values provided
    if param_values is None:
        if param_name == 'tau1':
            param_values = [100, 30, 8, 1]
        else:
            param_values = [1.0, 0.75, 0.5, 0.25]  # Default values for other parameters
    
    parameter_sets = []
    
    # Create parameter sets with different parameter values
    for value in param_values:
        params = {
            'tau1': 8,
            'tau2': 200,
            'tau3': 1,
            'tau4': np.inf,
            'k1': 0.74,
            'k2': 1.0,
            'k3': 0.07,
            'k4': 0.0312
        }
        # Update the specified parameter
        params[param_name] = value
        parameter_sets.append(params)

    # Create directory for CSV files if it doesn't exist
    csv_dir = "spike_data"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for vf in vf_tip_sizes:
        try:
            if vf == 4.56:
                data = pd.read_csv(f"data/P3/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv")
            else:
                data = pd.read_csv(f"data/P4/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv")
            logging.warning(f"Reading data for {vf}")
            time = data['Time (ms)'].to_numpy()
            stress = data[data.columns[1]].values
        except KeyError as e:
            logging.warning(f"File not found for {vf} and {ramp}")
            continue

        # Plot stress traces in black
        axs[0].plot(time, stress, label=f"VF {vf}mm", color='black')

        # Generate and plot spikes for each parameter set
        for (params, value, color) in zip(parameter_sets, param_values, parameter_colors):
            # Update parameters
            for param_name, param_value in params.items():
                lmpars[param_name].value = param_value

            groups = MC_GROUPS
            if afferent_type == "SA":
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=0.2, h=0.5)
            elif afferent_type == "RA":
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=0.4, h=1)

            if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
                logging.warning(f"SPIKES COULD NOT BE GENERATED for {param_name}={value}")
                continue

            if len(mod_spike_time) != len(mod_fr_inst):
                if len(mod_fr_inst) > 1:
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
                else:
                    mod_fr_inst_interp = np.zeros_like(mod_spike_time)
            else:
                mod_fr_inst_interp = mod_fr_inst

            # Save spike times and firing rates to CSV
            csv_filename = f"{csv_dir}/{afferent_type}_{ramp}_vf{vf}_{param_name}_{value}.csv"
            spike_data = pd.DataFrame({
                'spike_time_ms': mod_spike_time,
                'firing_rate_kHz': mod_fr_inst_interp * 1e3
            })
            spike_data.to_csv(csv_filename, index=False)
            logging.warning(f"Saved spike data to {csv_filename}")

            # Plot spikes with different colors and continuous lines
            param_label = param_name if param_name.startswith('tau') else f"k{param_name[-1]}"
            axs[1].plot(mod_spike_time, mod_fr_inst_interp * 1e3, 
                      color=color, label=f"{param_label} = {value}",
                      marker='o', markersize=4, linestyle='none')

    # Configure axes
    for ax in axs:
        ax.set_xlim(left=0)
        ax.minorticks_on()

    # Set specific y-axis bounds for each subplot
    axs[0].set_ylim(bottom=0, top=200)  # For stress traces
    axs[1].set_ylim(bottom=0, top=600)  # For firing rates

    axs[0].set_title(f"{afferent_type} Von Frey Stress Trace ({vf_tip_sizes[0]}mm)")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Stress (kPa)")
    axs[0].legend(loc="best")

    axs[1].set_title(f"IFF for Different {param_name} Values")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("IFF (Hz)")
    axs[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(f"vf_graphs/stress_iffs_different_plot/{afferent_type}_{ramp}_{param_name}_comparison.png")
    plt.savefig(f"Figure1/{afferent_type}_{ramp}_{param_name}_comparison.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot single unit model results.')
    parser.add_argument('afferent_type', choices=['SA', 'RA'], help='Type of afferent (SA or RA)')
    parser.add_argument('ramp', choices=['shallow', 'out', 'steep'], help='Type of ramp (shallow, out, or steep)')
    parser.add_argument('--scaling_factor', type=float, default=1.0, help='Scaling factor for stress values (default: 1.0)')
    parser.add_argument('--plot_type', choices=['single', 'parameter'], default='single', help='Plot type (single or parameter)')
    parser.add_argument('--param_name', choices=['tau1', 'tau2', 'tau3', 'tau4', 'k1', 'k2', 'k3', 'k4'], 
                       default='tau1', help='Parameter to vary in parameter comparison plot')
    parser.add_argument('--param_values', type=float, nargs='+', 
                       help='Space-separated list of values for the parameter (e.g., --param_values 100 30 8 1)')
    
    args = parser.parse_args()
    
    if args.plot_type == 'single':
        run_same_plot(
            args.afferent_type,
            args.ramp,
            scaling_factor=args.scaling_factor
        )
    elif args.plot_type == 'parameter':
        plot_parameter_comparison(
            args.afferent_type,
            args.ramp,
            param_name=args.param_name,
            param_values=args.param_values
        )

if __name__ == '__main__':
    main()
