import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import argparse
from scipy.ndimage import gaussian_filter1d

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plot parameters
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20  # Very big font

def plot_aligned_simplified(aligned_file, afferent_type="SA", scaling_factor=1.0, plot_style="smooth"):
    """
    Plot the stress traces and simplified firing rates from aligned stress traces.
    This function creates a simplified model that generates firing rates throughout the time range.
    
    Parameters:
    -----------
    aligned_file : str
        Path to the aligned stress traces CSV file.
    afferent_type : str, optional
        Type of afferent to simulate ("RA" or "SA"). Default is "SA".
    scaling_factor : float, optional
        Scaling factor for stress values. Default is 1.0.
    plot_style : str, optional
        Style of the firing rate plot ("smooth" or "points"). Default is "smooth".
    """
    # Load the aligned data
    aligned_data = pd.read_csv(aligned_file)
    
    # Extract time and VF tip sizes
    time = aligned_data['Time (ms)'].to_numpy()
    vf_columns = [col for col in aligned_data.columns if col.startswith('Stress_')]
    vf_tip_sizes = [float(col.split('_')[1]) for col in vf_columns]
    
    # Print the time range of the data
    logging.info(f"Time range in data: {time[0]} to {time[-1]} ms")
    
    # Colors from the original run_same_plot function
    colors = ['#440154', '#3b528b', '#21908c', '#5dc963', '#fde725']
    
    # Create the figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot each VF tip size
    for vf_col, color in zip(vf_columns, colors):
        vf = vf_col.split('_')[1]
        stress = aligned_data[vf_col].values
        
        # Print the stress range for debugging
        logging.info(f"VF {vf} stress range: {np.min(stress):.2f} to {np.max(stress):.2f} kPa")
        
        # Plot the stress trace
        axs[0].plot(time, stress, label=f"VF {vf}", color=color)
        
        # Calculate simplified firing rates that cover the full time range
        # Use a sigmoid function to model the relationship between stress and firing rate
        # This ensures we get firing rates throughout the entire time range
        
        # Parameters for SA type afferents
        if afferent_type == "SA":
            # SA afferents have persistent firing during sustained stimuli
            baseline = 5.0  # Baseline firing rate (Hz)
            max_rate = 200.0  # Maximum firing rate (Hz)
            sensitivity = 0.02  # Sensitivity to stress
            threshold = 50.0  # Threshold for activation
            
            # Calculate simplified firing rates (Hz)
            # Base rate dependent on stress level
            base_firing_rate = baseline + max_rate * (1 / (1 + np.exp(-(stress - threshold) * sensitivity)))
            
            # Add response to rate of change (first derivative) - captures on/off responses
            stress_derivative = np.gradient(stress, time)
            derivative_component = np.abs(stress_derivative) * 0.5  # Scale factor for derivative
            
            # Combine components
            firing_rate = base_firing_rate + derivative_component
            
            # Ensure rates are non-negative
            firing_rate = np.maximum(firing_rate, 0)
            
        else:  # RA type
            # RA afferents primarily respond to changes in stress
            baseline = 0.0  # Baseline firing rate (Hz)
            max_rate = 250.0  # Maximum firing rate (Hz)
            sensitivity = 0.5  # Sensitivity to stress changes
            
            # Calculate firing rates based mainly on stress derivative
            stress_derivative = np.gradient(stress, time)
            
            # Only positive derivatives for RA (mostly responds to increasing stress)
            stress_derivative = np.clip(stress_derivative, 0, None)
            
            # Calculate firing rate
            firing_rate = baseline + max_rate * (1 - np.exp(-sensitivity * stress_derivative))
            
            # Add small stress-dependent component to maintain some firing during sustained stimuli
            stress_component = 0.02 * stress  # Small direct contribution from stress level
            firing_rate += stress_component
            
            # Ensure rates are non-negative
            firing_rate = np.maximum(firing_rate, 0)
        
        # Apply smoothing if needed
        if plot_style == "smooth":
            # Smooth the firing rates
            firing_rate = gaussian_filter1d(firing_rate, sigma=15)
        
        # Create simulated spike times for the point plot option
        if plot_style == "points":
            # Generate spike times based on firing rate
            spike_times = []
            spike_rates = []
            
            for i in range(len(time)-1):
                # Consider generating a spike in this time interval
                dt = time[i+1] - time[i]
                rate = firing_rate[i]
                
                # Probability of a spike in this interval
                p_spike = rate * dt / 1000.0  # Convert ms to seconds
                
                if np.random.random() < p_spike and rate > 10:  # Only generate spikes for rates above 10 Hz
                    spike_times.append(time[i])
                    spike_rates.append(rate)
            
            # Plot as points
            axs[1].plot(spike_times, spike_rates, color=color, label=f"VF {vf}",
                      marker="o", linestyle="none")
            
            # Count spikes
            logging.info(f"VF {vf}: {len(spike_times)} simulated spikes")
            if len(spike_times) > 0:
                logging.info(f"Spike time range: {min(spike_times):.2f} to {max(spike_times):.2f} ms")
        else:
            # Plot as a continuous line
            axs[1].plot(time, firing_rate, color=color, label=f"VF {vf}", 
                      marker="", linestyle="solid")
            
            # Log the rate range
            logging.info(f"VF {vf} firing rate range: {np.min(firing_rate):.2f} to {np.max(firing_rate):.2f} Hz")
    
    # Configure both axes
    for ax in axs:
        ax.set_xlim(left=0, right=5000)  # Set x-axis to show the full range
        ax.set_ylim(bottom=0)  # Set y-axis to start at 0
        ax.minorticks_on()  # Enable minor ticks
        ax.grid(True, which='major', linestyle='-', alpha=0.3)  # Add major grid lines
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)  # Add minor grid lines
    
    # Add titles and labels
    axs[0].set_title(f"{afferent_type} Aligned Von Frey Stress Traces")
    axs[0].set_ylabel("Stress (kPa)")
    axs[0].legend(loc="best")
    
    axs[1].set_title(f"{afferent_type} Simplified Firing Rates")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Firing Rate (Hz)")
    axs[1].legend(loc="best")
    
    # Generate output file names
    base_name = os.path.basename(aligned_file)
    name_parts = os.path.splitext(base_name)[0]
    
    # Save the figure
    output_dir = "aligned_firing_simplified"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{name_parts}_{afferent_type}_{plot_style}_simplified.png")
    plt.tight_layout()
    plt.savefig(output_file)
    
    # Also save to Figure1 folder for consistency with original code
    os.makedirs("Figure1", exist_ok=True)
    plt.savefig(f"Figure1/{name_parts}_{afferent_type}_{plot_style}_simplified.png")
    
    logging.info(f"Figure saved to {output_file}")
    
    return fig, axs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot simplified firing rates from aligned stress traces.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the aligned stress traces CSV file')
    parser.add_argument('--afferent_type', type=str, default='SA', choices=['SA', 'RA'], help='Type of afferent (SA or RA)')
    parser.add_argument('--scaling_factor', type=float, default=1.0, help='Scaling factor for stress values')
    parser.add_argument('--plot_style', type=str, default='smooth', choices=['smooth', 'points'], help='Style of the firing rate plot')
    
    args = parser.parse_args()
    
    # Plot the firing rates
    fig, axs = plot_aligned_simplified(
        args.input_file,
        afferent_type=args.afferent_type,
        scaling_factor=args.scaling_factor,
        plot_style=args.plot_style
    )
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main() 
