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
lmpars.add('k3', value=.07, vary=False, min=0) #c constant
lmpars.add('k4', value=.0312, vary=False, min=0)
lmpars_init_dict['t3f12v3final'] = lmpars
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20  # Very big font

param_display_names = {
    'tau1': 'tauRI',
    'tau2': 'tauSI',
    'tau3': 'tauUSI',
    'tau4': 'tauVSI',
    'k1': 'k1',
    'k2': 'k2',
    'k3': 'k3',
    'k4': 'k4'
}

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
            axs[vf_idx, ramp_idx].plot(mod_spike_time, mod_fr_inst_interp * 1e3, label="IFF (Hz)", marker='o', linestyle='none')
            axs[vf_idx, ramp_idx].set_title(f"{vf} {ramp} {afferent_type} Afferent")
            axs[vf_idx, ramp_idx].set_ylabel('IFF (Hz) / Stress (kPa)')



            if legend_added is False:
                axs[vf_idx, ramp_idx].legend(loc='lower center', bbox_to_anchor=(0.5, 1.15))
                legend_added = True

    plt.tight_layout()
    plt.show()

'''
shallow, LIF_RESOLUTION == 2
out, LIF_RESOLUTION == 1
steep, LIF_RESOLUTION == .5
'''

def run_same_plot(afferent_type, ramp, scaling_factor = .1, plot_style = "smooth"):
    colors = ['#3D2674', '#6677FA', '#FF9047', '#92CA68']  # Updated colors
    vf_tip_sizes = [3.61, 4.08, 4.31, 4.56]
    LifConstants.set_resolution(1)

    # Define figure parameters
    fig_params = {
        'figsize': (14, 6),
        'left': 0.098,
        'right': 0.95,
        'top': 0.937,
        'bottom': 0.097
    }

    # Create figure with specified parameters
    fig = plt.figure(figsize=fig_params['figsize'])
    gs = fig.add_gridspec(1, 1,
                         left=fig_params['left'],
                         right=fig_params['right'],
                         top=fig_params['top'],
                         bottom=fig_params['bottom'])
    ax = fig.add_subplot(gs[0])

    for vf, color in zip(vf_tip_sizes, colors):
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
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=0.2, h=.5)
        elif afferent_type == "RA":
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=0.4, h=1)
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
            mod_spike_time = np.array(list(mod_spike_time) + [1001, 4001])
            mod_fr_inst_interp = np.array(list(mod_fr_inst_interp) + [0, 0])
            sort_idx = np.argsort(mod_spike_time)
            mod_spike_time = mod_spike_time[sort_idx]
            mod_fr_inst_interp = mod_fr_inst_interp[sort_idx]

        ax.plot(mod_spike_time, mod_fr_inst_interp * 1e3, color=color, label=f"{vf}", marker="o", markersize=4, linestyle="none")

    ax.set_xlim(0, 5000)  # Set x-axis limits from 0 to 5000
    ax.set_ylim(bottom=0, top=275)
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
    vf_tip_sizes = [3.61, 4.08, 4.31, 4.56]
    colors = ['#3D2674', '#6677FA', '#FF9047', '#92CA68']
    vf_list_len = len(vf_tip_sizes)

    # Create subplots for firing rate and stress in a 5x1 layout
    fig, axs = plt.subplots(vf_list_len, 1, figsize=(8, 10), sharex=True, sharey=True)

    legend_added = False
    for vf_idx, vf in enumerate(vf_tip_sizes):
        color = colors[vf_idx]
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

        axs[vf_idx].plot(mod_spike_time, mod_fr_inst_interp * 1e3, label="IFF (Hz)", marker='o', linestyle='none', color=color)
        axs[vf_idx].set_title(f"{vf} {ramp} {afferent_type} Afferent")
        axs[vf_idx].set_ylabel('IFF (Hz)')
        
        if not legend_added:
            axs[vf_idx].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
            legend_added = True

    fig.suptitle(f"Firing Rate and Stress for Ramp Type: {ramp}")
    plt.tight_layout()

def plot_parameter_comparison(afferent_type, ramp, param_name='tau1', param_values=None):
    # Ensure new save directory exists
    os.makedirs('vf_graphs/fig3parametersensitivity', exist_ok=True)
    # Colors from darkest to lightest gray
    parameter_colors = ['black', '#666666', '#999999', '#CCCCCC']  # From darkest to lightest
    vf_tip_sizes = [4.56]
    LifConstants.set_resolution(1)

    # Define consistent figure parameters
    fig_params = {
        'figsize': (14, 6),
        'left': 0.098,    # left margin
        'right': 0.95,    # right margin
        'top': 0.937,     # top margin
        'bottom': 0.097   # bottom margin
    }

    figs = []
    axs = []
    # Create two separate figures, each matching the idealized stress plot dimensions
    for i in range(2):
        fig = plt.figure(figsize=fig_params['figsize'])
        gs = fig.add_gridspec(1, 1,
                            left=fig_params['left'],
                            right=fig_params['right'],
                            top=fig_params['top'],
                            bottom=fig_params['bottom'])
        ax = fig.add_subplot(gs[0])
        figs.append(fig)
        axs.append(ax)

    # Set y-axis limits based on afferent type and subplot
    if afferent_type == "SA":
        top_y_lim = 275
        bottom_y_lim = 550
    else:  # RA
        top_y_lim = 275  # Changed from 550 to match SA
        bottom_y_lim = 550  # Changed from 275 to match SA

    # Define parameter values based on afferent type
    if afferent_type == "SA":
        tau1_values = [1, 8, 30, 100]
        tau2_values = [50, 200, 350, 500]  # TauSI
        tau3_values = [100, 500, 1000, 1744.6]  # TauUSI
        k2_values = [2, 4, 8, 16]  # b parameter for SA
        k3_values = [0.01, 0.07, 0.15, 0.3]  # c parameter for SA
    else:  # RA
        tau1_values = [1.0, 1.5, 2.0, 2.5]
        k1_values = [10, 20, 30, 40]  # a parameter for RA

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

        # Set base parameters based on afferent type
        if afferent_type == "SA":
            base_params = {
                'tau1': 8, 'tau2': 200, 'tau3': 1744.6, 'tau4': np.inf,
                'k1': 0.74, 'k2': 1.0, 'k3': 0.07, 'k4': 0.0312
            }
            g, h = 0.2, 0.5
        else:  # RA
            base_params = {
                'tau1': 2.5, 'tau2': 200, 'tau3': 1, 'tau4': np.inf,
                'k1': 35, 'k2': 0, 'k3': 0.0, 'k4': 0
            }
            g, h = 0.4, 1.0

        # Plot TauRI variations (first figure)
        for value, color in zip(tau1_values, parameter_colors):
            for param, val in base_params.items():
                lmpars[param].value = val
            lmpars['tau1'].value = value
            groups = MC_GROUPS
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=g, h=h)
            if len(mod_spike_time) > 0 and len(mod_fr_inst) > 0:
                if len(mod_spike_time) != len(mod_fr_inst):
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst) if len(mod_fr_inst) > 1 else np.zeros_like(mod_spike_time)
                else:
                    mod_fr_inst_interp = mod_fr_inst
                axs[0].plot(mod_spike_time, mod_fr_inst_interp * 1e3, 
                          color=color, label=f"τRI = {value}",
                          marker='o', markersize=4, linestyle="none")

        # Plot k2 (b) or k1 (a) variations (second figure)
        param_values = k2_values if afferent_type == "SA" else k1_values
        param_name = "k2" if afferent_type == "SA" else "k1"
        param_label = "b" if afferent_type == "SA" else "a"
        for value, color in zip(param_values, parameter_colors):
            for param, val in base_params.items():
                lmpars[param].value = val
            lmpars[param_name].value = value
            groups = MC_GROUPS
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=g, h=h)
            if len(mod_spike_time) > 0 and len(mod_fr_inst) > 0:
                if len(mod_spike_time) != len(mod_fr_inst):
                    mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst) if len(mod_fr_inst) > 1 else np.zeros_like(mod_spike_time)
                else:
                    mod_fr_inst_interp = mod_fr_inst
                axs[1].plot(mod_spike_time, mod_fr_inst_interp * 1e3, 
                          color=color, label=f"{param_label} = {value}",
                          marker='o', markersize=4, linestyle="none")

        # Additional plots for SA: tau2, tau3, k3
        if afferent_type == "SA":
            # tau2 (TauSI)
            fig_tau2 = plt.figure(figsize=fig_params['figsize'])
            gs_tau2 = fig_tau2.add_gridspec(1, 1,
                                          left=fig_params['left'],
                                          right=fig_params['right'],
                                          top=fig_params['top'],
                                          bottom=fig_params['bottom'])
            ax_tau2 = fig_tau2.add_subplot(gs_tau2[0])
            for value, color in zip(tau2_values, parameter_colors):
                for param, val in base_params.items():
                    lmpars[param].value = val
                lmpars['tau2'].value = value
                groups = MC_GROUPS
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=g, h=h)
                if len(mod_spike_time) > 0 and len(mod_fr_inst) > 0:
                    if len(mod_spike_time) != len(mod_fr_inst):
                        mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst) if len(mod_fr_inst) > 1 else np.zeros_like(mod_spike_time)
                    else:
                        mod_fr_inst_interp = mod_fr_inst
                    ax_tau2.plot(mod_spike_time, mod_fr_inst_interp * 1e3, color=color, label=f"τSI = {value}", marker='o', markersize=4, linestyle="none")
            ax_tau2.set_xlim(0, 5000)
            ax_tau2.set_ylim(bottom=0, top=top_y_lim)
            # ax_tau2.set_xlabel("Time (ms)")  # Removing Time (ms) label
            ax_tau2.set_ylabel("IFF (Hz)")
            ax_tau2.set_title(f"IFF for Different τSI Values (SA)")
            ax_tau2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
            fig_tau2.savefig(f"vf_graphs/fig3parametersensitivity/SA_{ramp}_tau2_comparison.png", bbox_inches='tight', dpi=300)
            fig_tau2.savefig(f"Figure1/SA_{ramp}_tau2_comparison.png", bbox_inches='tight', dpi=300)

            # tau3 (TauUSI)
            fig_tau3 = plt.figure(figsize=fig_params['figsize'])
            gs_tau3 = fig_tau3.add_gridspec(1, 1,
                                          left=fig_params['left'],
                                          right=fig_params['right'],
                                          top=fig_params['top'],
                                          bottom=fig_params['bottom'])
            ax_tau3 = fig_tau3.add_subplot(gs_tau3[0])
            for value, color in zip(tau3_values, parameter_colors):
                for param, val in base_params.items():
                    lmpars[param].value = val
                lmpars['tau3'].value = value
                groups = MC_GROUPS
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=g, h=h)
                if len(mod_spike_time) > 0 and len(mod_fr_inst) > 0:
                    if len(mod_spike_time) != len(mod_fr_inst):
                        mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst) if len(mod_fr_inst) > 1 else np.zeros_like(mod_spike_time)
                    else:
                        mod_fr_inst_interp = mod_fr_inst
                    ax_tau3.plot(mod_spike_time, mod_fr_inst_interp * 1e3, color=color, label=f"τUSI = {value}", marker='o', markersize=4, linestyle="none")
            ax_tau3.set_xlim(0, 5000)
            ax_tau3.set_ylim(bottom=0, top=top_y_lim)
            # ax_tau3.set_xlabel("Time (ms)")  # Removing Time (ms) label
            ax_tau3.set_ylabel("IFF (Hz)")
            ax_tau3.set_title(f"IFF for Different τUSI Values (SA)")
            ax_tau3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
            fig_tau3.savefig(f"vf_graphs/fig3parametersensitivity/SA_{ramp}_tau3_comparison.png", bbox_inches='tight', dpi=300)
            fig_tau3.savefig(f"Figure1/SA_{ramp}_tau3_comparison.png", bbox_inches='tight', dpi=300)

            # k3 (c)
            fig_k3 = plt.figure(figsize=fig_params['figsize'])
            gs_k3 = fig_k3.add_gridspec(1, 1,
                                      left=fig_params['left'],
                                      right=fig_params['right'],
                                      top=fig_params['top'],
                                      bottom=fig_params['bottom'])
            ax_k3 = fig_k3.add_subplot(gs_k3[0])
            for value, color in zip(k3_values, parameter_colors):
                for param, val in base_params.items():
                    lmpars[param].value = val
                lmpars['k3'].value = value
                groups = MC_GROUPS
                mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=g, h=h)
                if len(mod_spike_time) > 0 and len(mod_fr_inst) > 0:
                    if len(mod_spike_time) != len(mod_fr_inst):
                        mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst) if len(mod_fr_inst) > 1 else np.zeros_like(mod_spike_time)
                    else:
                        mod_fr_inst_interp = mod_fr_inst
                    ax_k3.plot(mod_spike_time, mod_fr_inst_interp * 1e3, color=color, label=f"c = {value}", marker='o', markersize=4, linestyle="none")
            ax_k3.set_xlim(0, 5000)
            ax_k3.set_ylim(bottom=0, top=top_y_lim)
            # ax_k3.set_xlabel("Time (ms)")  # Removing Time (ms) label
            ax_k3.set_ylabel("IFF (Hz)")
            ax_k3.set_title(f"IFF for Different c Values (SA)")
            ax_k3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
            fig_k3.savefig(f"vf_graphs/fig3parametersensitivity/SA_{ramp}_k3_comparison.png", bbox_inches='tight', dpi=300)
            fig_k3.savefig(f"Figure1/SA_{ramp}_k3_comparison.png", bbox_inches='tight', dpi=300)

    # Configure axes
    for i, ax in enumerate(axs):
        ax.set_xlim(0, 5000)  # For firing frequency plot
        ax.minorticks_on()
        # ax.set_xlabel("Time (ms)")  # Removing Time (ms) label
        # Set different y-limits for each subplot
        ax.set_ylim(bottom=0, top=top_y_lim if i == 0 else bottom_y_lim)

    # Set titles and labels
    axs[0].set_title(f"IFF for Different τRI Values ({afferent_type})")
    axs[1].set_title(f"IFF for Different {'b' if afferent_type == 'SA' else 'a'} Values ({afferent_type})")
    for ax in axs:
        ax.set_ylabel("IFF (Hz)")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))

    # Save each figure separately with consistent parameters
    for i, fig in enumerate(figs):
        fig.savefig(f"vf_graphs/fig3parametersensitivity/{afferent_type}_{ramp}_{'tau1' if i == 0 else 'b' if afferent_type == 'SA' else 'a'}_comparison.png", 
                   bbox_inches='tight', dpi=300)
        fig.savefig(f"Figure1/{afferent_type}_{ramp}_{'tau1' if i == 0 else 'b' if afferent_type == 'SA' else 'a'}_comparison.png", 
                   bbox_inches='tight', dpi=300)
    plt.show()

def plot_stress_trace(ramp):
    """Plot just the stress trace for 3.61mm von Frey tip."""
    vf = 3.61
    try:
        if vf == 4.56:
            data = pd.read_csv(f"data/P3/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv")
        else:
            data = pd.read_csv(f"data/P4/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv")
        logging.warning(f"Reading data for {vf}")
        time = data['Time (ms)'].to_numpy()
        stress = data[data.columns[1]].values

        # Create figure with exact spacing parameters
        fig = plt.figure(figsize=(12, 5))  # Same size as parameter plots
        
        # Create subplot with exact spacing parameters
        gs = fig.add_gridspec(1, 1, 
                            left=0.098,    # left margin
                            right=0.95,    # right margin
                            top=0.937,     # top margin
                            bottom=0.097)  # bottom margin
        ax = fig.add_subplot(gs[0])
        
        # Plot stress trace in black
        ax.plot(time, stress, color='black')
        
        # Configure axis
        ax.set_xlim(0, 5000)  # For stress plot
        ax.set_ylim(bottom=0, top=200)  # Match y-axis limit from parameter plots
        ax.minorticks_on()
        
        # Set labels
        # ax.set_xlabel("Time (ms)")  # Removing Time (ms) label
        ax.set_ylabel("Stress (kPa)")
        ax.set_title(f"Stress Trace for {vf}mm von Frey")
        
        plt.savefig(f"vf_graphs/stress_trace/stress_{vf}_{ramp}.png")
        plt.savefig(f"Figure1/stress_{vf}_{ramp}.png")
        plt.show()

    except KeyError as e:
        logging.warning(f"File not found for {vf} and {ramp}")
        return

def plot_firing_rate_only(afferent_type, ramp):
    """
    Plot only the firing rate (IFF) for each von Frey size for a given ramp and afferent type.
    This is similar to run_single_unit_model_combined_graph but omits the stress trace.
    """
    vf_tip_sizes = [3.61, 4.08, 4.31, 4.56]  # The five tip sizes to plot
    vf_list_len = len(vf_tip_sizes)
    colors = ['#3D2674', '#6677FA', '#FF9047', '#92CA68']  # Color set for plotting

    # Create subplots for firing rate in a 5x1 layout
    fig, axs = plt.subplots(vf_list_len, 1, figsize=(8, 10), sharex=True, sharey=True)

    legend_added = False
    for vf_idx, vf in enumerate(vf_tip_sizes):
        color = colors[vf_idx]
        data = pd.read_csv(f"data/vf_unscaled/{vf}_{ramp}.csv")
        time = data['Time (ms)'].to_numpy()

        # Scaling factor
        scaling_factor = 0.28
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
            mod_spike_time = np.array(list(mod_spike_time) + [1001, 3999])
            mod_fr_inst_interp = np.array(list(mod_fr_inst_interp) + [0, 0])
            sort_idx = np.argsort(mod_spike_time)
            mod_spike_time = mod_spike_time[sort_idx]
            mod_fr_inst_interp = mod_fr_inst_interp[sort_idx]

        axs[vf_idx].plot(mod_spike_time, mod_fr_inst_interp * 1e3, label="IFF (Hz)", marker='o', linestyle='none', color=color)
        axs[vf_idx].set_title(f"{vf} {ramp} {afferent_type} Afferent")
        axs[vf_idx].set_ylabel('IFF (Hz)')
        
        if not legend_added:
            axs[vf_idx].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
            legend_added = True

    fig.suptitle(f"Firing Rate Only for Ramp Type: {ramp}")
    plt.tight_layout()
    plt.show()

def plot_

def main():
    parser = argparse.ArgumentParser(description='Plot single unit model results.')
    parser.add_argument('afferent_type', choices=['SA', 'RA'], help='Type of afferent (SA or RA)')
    parser.add_argument('ramp', choices=['shallow', 'out', 'steep'], help='Type of ramp (shallow, out, or steep)')
    parser.add_argument('--scaling_factor', type=float, default=1.0, help='Scaling factor for stress values (default: 1.0)')
    parser.add_argument('--plot_type', choices=['single', 'parameter', 'stress'], default='single', 
                      help='Plot type (single, parameter, or stress)')
    parser.add_argument('--param_name', choices=['tau1', 'tau2', 'tau3', 'tau4', 'k1', 'k2', 'k3', 'k4'], 
                       default='tau1', help='Parameter to vary in parameter comparison plot')
    parser.add_argument('--param_values', type=float, nargs='+', 
                       help='Space-separated list of values for the parameter (e.g., --param_values 100 30 8 1)')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("vf_graphs/stress_trace", exist_ok=True)
    
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
    elif args.plot_type == 'stress':
        plot_stress_trace(args.ramp)
    elif args.plot_type == 'firing_rate_only':
        plot_firing_rate_only(args.afferent_type, args.ramp)

if __name__ == '__main__':
    main()
