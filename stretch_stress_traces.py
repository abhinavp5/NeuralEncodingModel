import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_inflection_points(time, stress):
    """
    Find the ramp-up and ramp-down inflection points in a stress trace.
    
    Parameters:
    -----------
    time : array
        Time axis.
    stress : array
        Stress values.
    
    Returns:
    --------
    tuple
        (ramp_up_end_idx, ramp_down_start_idx)
    """
    # Calculate the first derivative
    d_stress = np.gradient(stress, time)
    
    # Calculate the second derivative
    dd_stress = np.gradient(d_stress, time)
    
    # Find positive and negative peaks in the second derivative
    # These correspond to the inflection points where the curve changes from ramp to steady and vice versa
    pos_peaks, _ = find_peaks(dd_stress, height=0.05*np.max(np.abs(dd_stress)))
    neg_peaks, _ = find_peaks(-dd_stress, height=0.05*np.max(np.abs(dd_stress)))
    
    # Filter peaks to focus on the main transitions
    # Typically, the ramp-up ends around ~1000ms and ramp-down starts around ~4000ms
    ramp_up_candidates = [idx for idx in pos_peaks if time[idx] < 1500]
    ramp_down_candidates = [idx for idx in neg_peaks if time[idx] > 3500]
    
    if not ramp_up_candidates or not ramp_down_candidates:
        # Fall back to a simple threshold approach if peak detection fails
        stress_range = np.max(stress) - np.min(stress)
        stress_threshold = np.min(stress) + 0.8 * stress_range  # 80% of max stress
        
        steady_indices = np.where(stress >= stress_threshold)[0]
        if len(steady_indices) > 0:
            ramp_up_end_idx = steady_indices[0]
            ramp_down_start_idx = steady_indices[-1]
        else:
            # Fallback to fixed points if all else fails
            ramp_up_end_idx = int(len(time) * 0.2)  # Assume ramp-up ends at 20% of the time
            ramp_down_start_idx = int(len(time) * 0.8)  # Assume ramp-down starts at 80% of the time
    else:
        ramp_up_end_idx = ramp_up_candidates[-1]  # Last peak in the ramp-up phase
        ramp_down_start_idx = ramp_down_candidates[0]  # First peak in the ramp-down phase
    
    return ramp_up_end_idx, ramp_down_start_idx

def stretch_stress_trace(time, stress, target_ramp_up_end, target_ramp_down_start):
    """
    Stretch the steady-state portion of a stress trace to match target inflection points.
    
    Parameters:
    -----------
    time : array
        Original time axis.
    stress : array
        Original stress values.
    target_ramp_up_end : float
        Target time for the end of ramp-up phase.
    target_ramp_down_start : float
        Target time for the start of ramp-down phase.
    
    Returns:
    --------
    tuple
        (new_time, new_stress)
    """
    # Find the current inflection points
    ramp_up_end_idx, ramp_down_start_idx = find_inflection_points(time, stress)
    
    ramp_up_end_time = time[ramp_up_end_idx]
    ramp_down_start_time = time[ramp_down_start_idx]
    
    logging.info(f"Inflection points: ramp_up_end={ramp_up_end_time:.2f}ms, ramp_down_start={ramp_down_start_time:.2f}ms")
    
    # Create a mapping from old time to new time
    new_time = np.zeros_like(time)
    
    # Keep the ramp-up phase unchanged
    new_time[:ramp_up_end_idx] = time[:ramp_up_end_idx]
    
    # Stretch the steady-state phase
    old_steady_state_duration = ramp_down_start_time - ramp_up_end_time
    new_steady_state_duration = target_ramp_down_start - target_ramp_up_end
    
    if old_steady_state_duration > 0:
        steady_state_stretch_factor = new_steady_state_duration / old_steady_state_duration
        
        # Apply stretching to the steady-state phase
        for i in range(ramp_up_end_idx, ramp_down_start_idx):
            relative_position = (time[i] - ramp_up_end_time) / old_steady_state_duration
            new_time[i] = target_ramp_up_end + relative_position * new_steady_state_duration
    
    # Adjust the ramp-down phase
    time_shift = target_ramp_down_start - ramp_down_start_time
    new_time[ramp_down_start_idx:] = time[ramp_down_start_idx:] + time_shift
    
    # Create an interpolation function to get stress values at the new time points
    interp_func = interp1d(time, stress, bounds_error=False, fill_value="extrapolate")
    
    # Create a new time array with uniform spacing
    # Find the start and end times of the new_time array
    start_time = new_time[0]
    end_time = new_time[-1]
    
    # Create a uniformly spaced time array
    uniform_time = np.linspace(start_time, end_time, len(time))
    
    # Interpolate stress values to the uniform time grid
    uniform_stress = interp_func(uniform_time)
    
    return uniform_time, uniform_stress

def process_stress_traces(input_file, output_file=None, target_ramp_up_end=1000, target_ramp_down_start=4000):
    """
    Process the stress traces in the input file to stretch the steady-state portion.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file with aggregated stress traces.
    output_file : str, optional
        Path to the output CSV file. If None, a default name is generated.
    target_ramp_up_end : float, optional
        Target time for the end of ramp-up phase in ms. Default is 1000ms.
    target_ramp_down_start : float, optional
        Target time for the start of ramp-down phase in ms. Default is 4000ms.
    
    Returns:
    --------
    str
        Path to the output CSV file.
    """
    # Read the input CSV file
    data = pd.read_csv(input_file)
    
    # Extract the time column
    original_time = data['Time (ms)'].values
    
    # Create a new DataFrame for the stretched traces
    stretched_data = pd.DataFrame()
    
    # Process each stress trace
    for col in data.columns:
        if col == 'Time (ms)':
            continue
        
        logging.info(f"Processing {col}...")
        
        # Extract the stress trace
        stress = data[col].values
        
        # Stretch the stress trace
        new_time, new_stress = stretch_stress_trace(
            original_time, 
            stress, 
            target_ramp_up_end, 
            target_ramp_down_start
        )
        
        # Add the stretched trace to the new DataFrame
        if 'Time (ms)' not in stretched_data:
            stretched_data['Time (ms)'] = new_time
        
        stretched_data[col] = new_stress
    
    # Generate the output file name if not provided
    if output_file is None:
        dir_name = os.path.dirname(input_file)
        base_name = os.path.basename(input_file)
        name_parts = os.path.splitext(base_name)
        output_file = os.path.join(dir_name, f"{name_parts[0]}_stretched{name_parts[1]}")
    
    # Save the stretched traces
    stretched_data.to_csv(output_file, index=False)
    logging.info(f"Stretched traces saved to {output_file}")
    
    return output_file

def plot_comparison(original_file, stretched_file, output_file=None):
    """
    Create a comparison plot between original and stretched stress traces.
    
    Parameters:
    -----------
    original_file : str
        Path to the original CSV file.
    stretched_file : str
        Path to the stretched CSV file.
    output_file : str, optional
        Path to the output plot file. If None, a default name is generated.
    
    Returns:
    --------
    str
        Path to the output plot file.
    """
    # Read the CSV files
    original_data = pd.read_csv(original_file)
    stretched_data = pd.read_csv(stretched_file)
    
    # Create a figure with multiple subplots (one for each VF tip size)
    vf_sizes = [col.split('_')[1] for col in original_data.columns if col != 'Time (ms)']
    num_vf = len(vf_sizes)
    
    fig, axes = plt.subplots(num_vf, 1, figsize=(12, 3*num_vf), sharex=True)
    
    # If there's only one VF tip size, axes will not be an array
    if num_vf == 1:
        axes = [axes]
    
    # Colors
    colors = {'original': 'blue', 'stretched': 'red'}
    
    # Plot each VF tip size
    for i, vf in enumerate(vf_sizes):
        col = f"Stress_{vf}"
        
        # Original trace
        axes[i].plot(
            original_data['Time (ms)'], 
            original_data[col], 
            label='Original', 
            color=colors['original']
        )
        
        # Stretched trace
        axes[i].plot(
            stretched_data['Time (ms)'], 
            stretched_data[col], 
            label='Stretched', 
            color=colors['stretched']
        )
        
        # Add vertical lines at target inflection points
        axes[i].axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
        axes[i].axvline(x=4000, color='gray', linestyle='--', alpha=0.5)
        
        # Add labels
        axes[i].set_title(f"VF {vf}")
        axes[i].set_ylabel("Stress (kPa)")
        axes[i].grid(True, linestyle='--', alpha=0.3)
        axes[i].legend()
    
    # Add a common x-label
    axes[-1].set_xlabel("Time (ms)")
    
    # Add an overall title
    fig.suptitle("Comparison of Original and Stretched Stress Traces", fontsize=16)
    
    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave some space for the title
    
    # Generate the output file name if not provided
    if output_file is None:
        dir_name = os.path.dirname(original_file)
        base_name = os.path.basename(original_file)
        name_parts = os.path.splitext(base_name)
        output_file = os.path.join(dir_name, f"{name_parts[0]}_comparison.png")
    
    # Save the figure
    plt.savefig(output_file)
    logging.info(f"Comparison plot saved to {output_file}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_file

def main():
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Stretch stress traces to align inflection points.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file')
    parser.add_argument('--ramp_up_end', type=float, default=1000, help='Target time for the end of ramp-up phase (ms)')
    parser.add_argument('--ramp_down_start', type=float, default=4000, help='Target time for the start of ramp-down phase (ms)')
    parser.add_argument('--no_plot', action='store_true', help='Skip creating a comparison plot')
    
    args = parser.parse_args()
    
    # Process the stress traces
    output_file = process_stress_traces(
        args.input_file,
        args.output_file,
        args.ramp_up_end,
        args.ramp_down_start
    )
    
    if output_file and not args.no_plot:
        # Create a comparison plot
        plot_file = plot_comparison(args.input_file, output_file)
        print(f"Comparison plot saved to {plot_file}")
    
    print(f"Stretched traces saved to {output_file}")

if __name__ == "__main__":
    main() 
