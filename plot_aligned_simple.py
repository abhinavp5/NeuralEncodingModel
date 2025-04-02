import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import argparse
from scipy.ndimage import gaussian_filter1d
from lmfit import minimize, fit_report, Parameters
from aim2_population_model_spatial_aff_parallel import get_mod_spike
from model_constants import MC_GROUPS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


# Set plot parameters
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20  # Very big font



def plot_aligned_simplified(aligned_file, afferent_type="SA", plot_style="smooth"):
    """
    Plot the stress traces and simplified firing rates from aligned stress traces.
    Uses the same smoothing logic as single_unit_plots.py.
    """
    # Load the aligned data with more robust parsing
    try:
        # First try with default settings
        aligned_data = pd.read_csv(aligned_file)
    except Exception as e:
        logging.error(f"Error loading CSV file: {str(e)}")
        raise FileExistsError(f"Error loading CSV file: {str(e)}")
    

    # Get the time values from the first column
    time = aligned_data.iloc[:, 0].values
    


    vf_tip_sizes = [3.61, 4.08, 4.17, 4.31, 4.56]  # Sorted in increasing order
    colors = ['#440154', '#3b528b', '#21908c', '#5dc963', '#fde725']

    # Create the figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    


    # Plot each VF tip size
    for idx, (vf_tip_size, color) in enumerate(zip(vf_tip_sizes, colors)):
        # Get the actual stress values from the appropriate column
        stress = aligned_data.iloc[:, idx+1].values
        logging.info(f"Stress values shape: {stress.shape}")
        logging.info(f"Time values shape: {time.shape}")
        # Set up model parameters based on afferent type

        if afferent_type == "SA":
            lmpars = lmpars_init_dict['t3f12v3final']
            lmpars['tau1'].value = 8
            lmpars['tau2'].value = 200
            lmpars['tau3'].value = 1 
            lmpars['tau4'].value = np.inf
            lmpars['k1'].value = 0.74
            lmpars['k2'].value = 1.0
            lmpars['k3'].value = 0.07
            lmpars['k4'].value = 0.0312
        elif afferent_type == "RA":
            lmpars = lmpars_init_dict['t3f12v3final']
            lmpars['tau1'].value = 2.5
            lmpars['tau2'].value = 200
            lmpars['tau3'].value = 1
            lmpars['k1'].value = 35
            lmpars['k2'].value = 0
            lmpars['k3'].value = 0.0
            lmpars['k4'].value = 0
        
        groups = MC_GROUPS
        if afferent_type == "SA":
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=0.2, h=0.5)
        elif afferent_type == "RA":
            mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress, g=0.4, h=1)
        
        logging.warning(f"VF: {vf_tip_size} - Generated {len(mod_spike_time)} spikes")
        if len(mod_spike_time) == 0 or len(mod_fr_inst) == 0:
            logging.warning(f"SPIKES COULD NOT BE GENERATED for VF {vf_tip_size}")
            continue
        if len(mod_spike_time) != len(mod_fr_inst):
            if len(mod_fr_inst) > 1:
                mod_fr_inst_interp = np.interp(mod_spike_time, time, mod_fr_inst)
            else:
                mod_fr_inst_interp = np.zeros_like(mod_spike_time)
        else:
            mod_fr_inst_interp = mod_fr_inst

        # Plot stress traces in top subplot
        axs[0].plot(time, stress, label=f"VF {vf_tip_size}", color=color)
        
        # Plot firing rates with updated smoothing logic
        if plot_style == "points":
            # Plot individual points without smoothing
            axs[1].plot(mod_spike_time, mod_fr_inst_interp * 1e3, 
                       color=color, label=f"VF {vf_tip_size}",
                       marker="o", markersize=4, linestyle="none")
        
        elif plot_style == "smooth":
            # First plot points with smaller markers
            axs[1].plot(mod_spike_time, mod_fr_inst_interp * 1e3, 
                       color=color, alpha=0.3,
                       marker="o", markersize=3, linestyle="none")
            
            # Define the steady state region
            steady_state_mask = ((mod_spike_time >= 1000) & (mod_spike_time <= 4000))
            
            # Create mask for points with actual firing during steady state
            firing_mask = mod_fr_inst_interp > 0
            
            # Identify regions of no firing in steady state
            no_firing_steady_state = steady_state_mask & ~firing_mask
            
            # Create smoothing for dynamic regions
            dynamic_regions = ~steady_state_mask
            smooth_rapid = gaussian_filter1d(mod_fr_inst_interp, sigma=3)
            
            # Combine based on masks
            final_fr = mod_fr_inst_interp.copy()
            
            # Apply smoothing only to dynamic regions
            final_fr[dynamic_regions] = smooth_rapid[dynamic_regions]
            
            # Set steady state regions with no firing to zero
            final_fr[no_firing_steady_state] = 0
            
            # Add smooth transitions at boundaries
            transition_points = np.where(np.diff(no_firing_steady_state.astype(int)) != 0)[0]
            window_size = 5  # Size of transition window
            
            for tp in transition_points:
                if tp > window_size and tp < len(final_fr) - window_size:
                    if final_fr[tp+1] == 0:  # Transitioning to zero
                        transition = np.linspace(final_fr[tp-window_size], 0, window_size*2)
                        final_fr[tp-window_size:tp+window_size] = transition
                    else:  # Transitioning from zero
                        transition = np.linspace(0, final_fr[tp+window_size], window_size*2)
                        final_fr[tp-window_size:tp+window_size] = transition
            
            # Plot the smoothed line
            axs[1].plot(mod_spike_time, final_fr * 1e3, 
                       color=color, label=f"VF {vf_tip_size}", 
                       linewidth=1.5)
    
    # Configure axes
    for ax in axs:
        ax.set_xlim(left=0)
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Set specific y-axis bounds
    axs[0].set_ylim(bottom=0, top=400)  # For stress traces
    axs[1].set_ylim(bottom=0, top=275)  # For firing rates
    
    # Set titles and labels
    axs[0].set_title(f"{afferent_type} Von Frey Stress Traces")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Stress (kPa)")
    axs[0].legend(loc="best")
    
    axs[1].set_title(f"{afferent_type} IFF's associated with Stress Traces")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Firing Rate (Hz)")
    axs[1].legend(loc="best")
    
    plt.tight_layout()
    return fig, axs

def plot_stress_traces_only(aligned_file):
    """
    Plot only the stress traces from aligned stress traces CSV file.
    
    Parameters:
    -----------
    aligned_file : str
        Path to the aligned stress traces CSV file.
    """
    # Load the aligned data
    try:
        aligned_data = pd.read_csv(aligned_file)
    except Exception as e:
        logging.error(f"Error loading CSV file: {str(e)}")
        raise FileExistsError(f"Error loading CSV file: {str(e)}")
    
    # Get the time values from the first column
    time = aligned_data.iloc[:, 0].values
    
    # Define colors for different traces
    colors = ['#440154', '#3b528b', '#21908c', '#5dc963', '#fde725']
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each stress trace
    for i in range(1, min(len(aligned_data.columns), len(colors)+1)):
        stress = aligned_data.iloc[:, i].values
        label = f"VF Tip {i}" if i < len(aligned_data.columns) else ""
        ax.plot(time, stress, label=label, color=colors[i-1])
    
    # Set plot labels and title
    ax.set_title("Stress Traces")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Stress (kPa)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig, ax

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot simplified firing rates from aligned stress traces.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the aligned stress traces CSV file')
    parser.add_argument('--afferent_type', type=str, default='RA', choices=['SA', 'RA'], help='Type of afferent (SA or RA)')
    parser.add_argument('--plot_style', type=str, default='points', choices=['smooth', 'points'], help='Style of the firing rate plot')
    parser.add_argument('--stress_only', action='store_true', help='Plot only stress traces without firing rates')
    
    args = parser.parse_args()
    
    # Choose which plotting function to use based on the arguments
    if args.stress_only:
        # Plot stress traces only
        fig, ax = plot_stress_traces_only(args.input_file)
    else:
        # Plot both stress traces and firing rates
        fig, axs = plot_aligned_simplified(
            args.input_file,
            afferent_type=args.afferent_type,
            plot_style=args.plot_style
        )
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main() 
