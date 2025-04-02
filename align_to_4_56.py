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

def align_to_reference_trace(input_file, output_file=None, reference_vf='4.56'):
    """
    Align all stress traces to the reference trace (4.56).
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file.
    output_file : str, optional
        Path to the output CSV file. If None, a default name is generated.
    reference_vf : str, optional
        VF tip size to use as reference. Default is '4.56'.
    
    Returns:
    --------
    str
        Path to the output CSV file.
    """
    # Read the input file
    data = pd.read_csv(input_file)
    
    # Extract the time column
    time = data['Time (ms)'].values
    
    # Get the reference stress trace
    reference_col = f"Stress_{reference_vf}"
    if reference_col not in data.columns:
        raise ValueError(f"Reference column {reference_col} not found in the input file")
    
    reference_stress = data[reference_col].values
    
    # Find inflection points for the reference trace
    ref_ramp_up_idx, ref_ramp_down_idx = find_inflection_points(time, reference_stress)
    ref_ramp_up_time = time[ref_ramp_up_idx]
    ref_ramp_down_time = time[ref_ramp_down_idx]
    
    logging.info(f"Reference trace ({reference_vf}) inflection points: ramp_up={ref_ramp_up_time:.2f}ms, ramp_down={ref_ramp_down_time:.2f}ms")
    
    # Create a new DataFrame for the aligned traces
    aligned_data = pd.DataFrame()
    aligned_data['Time (ms)'] = time
    
    # Add the reference trace unchanged
    aligned_data[reference_col] = reference_stress
    
    # Process each non-reference trace
    for col in data.columns:
        if col == 'Time (ms)' or col == reference_col:
            continue
        
        vf_size = col.split('_')[1]
        logging.info(f"Aligning {vf_size} to {reference_vf}...")
        
        # Extract the stress trace
        stress = data[col].values
        
        # Find inflection points for this trace
        ramp_up_idx, ramp_down_idx = find_inflection_points(time, stress)
        ramp_up_time = time[ramp_up_idx]
        ramp_down_time = time[ramp_down_idx]
        
        logging.info(f"  {vf_size} inflection points: ramp_up={ramp_up_time:.2f}ms, ramp_down={ramp_down_time:.2f}ms")
        
        # Create a mapping from old time to new time
        new_time = np.zeros_like(time)
        
        # Phase 1: Map ramp-up phase
        if ramp_up_idx > 0:
            # Scale the ramp-up phase to match reference
            for i in range(ramp_up_idx):
                relative_position = time[i] / ramp_up_time
                new_time[i] = relative_position * ref_ramp_up_time
        
        # Phase 2: Map steady-state phase
        old_steady_duration = ramp_down_time - ramp_up_time
        new_steady_duration = ref_ramp_down_time - ref_ramp_up_time
        
        if old_steady_duration > 0:
            for i in range(ramp_up_idx, ramp_down_idx):
                relative_position = (time[i] - ramp_up_time) / old_steady_duration
                new_time[i] = ref_ramp_up_time + relative_position * new_steady_duration
        
        # Phase 3: Map ramp-down phase
        if ramp_down_idx < len(time) - 1:
            remaining_time = time[-1] - ramp_down_time
            ref_remaining_time = time[-1] - ref_ramp_down_time
            
            if remaining_time > 0:
                for i in range(ramp_down_idx, len(time)):
                    relative_position = (time[i] - ramp_down_time) / remaining_time
                    new_time[i] = ref_ramp_down_time + relative_position * ref_remaining_time
        
        # Create an interpolation function to get stress values at the original time points
        interp_func = interp1d(new_time, stress, bounds_error=False, fill_value="extrapolate")
        
        # Interpolate stress values to match the original time grid
        aligned_stress = interp_func(time)
        
        # Add to the aligned data
        aligned_data[col] = aligned_stress
    
    # Generate output filename if not provided
    if output_file is None:
        dir_name = os.path.dirname(input_file)
        base_name = os.path.basename(input_file)
        name_parts = os.path.splitext(base_name)
        output_file = os.path.join(dir_name, f"{name_parts[0]}_aligned_to_{reference_vf}{name_parts[1]}")
    
    # Save the aligned data
    aligned_data.to_csv(output_file, index=False)
    logging.info(f"Aligned traces saved to {output_file}")
    
    return output_file

def plot_comparison(original_file, aligned_file, output_file=None, reference_vf='4.56'):
    """
    Create a comparison plot of original and aligned stress traces.
    
    Parameters:
    -----------
    original_file : str
        Path to the original CSV file.
    aligned_file : str
        Path to the aligned CSV file.
    output_file : str, optional
        Path to the output plot file. If None, a default name is generated.
    reference_vf : str, optional
        VF tip size used as reference. Default is '4.56'.
    
    Returns:
    --------
    str
        Path to the output plot file.
    """
    # Read the CSV files
    original_data = pd.read_csv(original_file)
    aligned_data = pd.read_csv(aligned_file)
    
    # Create a figure with two subplots (original and aligned)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Set titles
    axes[0].set_title("Original Stress Traces", fontsize=14)
    axes[1].set_title("Aligned Stress Traces", fontsize=14)
    
    # Get VF tip sizes and corresponding colors
    vf_sizes = [col.split('_')[1] for col in original_data.columns if col != 'Time (ms)']
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(vf_sizes)))
    
    # Plot original traces
    for i, vf in enumerate(vf_sizes):
        col = f"Stress_{vf}"
        line_width = 2.0 if vf == reference_vf else 1.5
        
        axes[0].plot(
            original_data['Time (ms)'], 
            original_data[col], 
            label=f"VF {vf}", 
            color=colors[i],
            linestyle='-',  # All lines solid
            linewidth=line_width
        )
    
    # Plot aligned traces
    for i, vf in enumerate(vf_sizes):
        col = f"Stress_{vf}"
        line_width = 2.0 if vf == reference_vf else 1.5
        
        axes[1].plot(
            aligned_data['Time (ms)'], 
            aligned_data[col], 
            label=f"VF {vf}", 
            color=colors[i],
            linestyle='-',  # All lines solid
            linewidth=line_width
        )
    
    # Configure all axes
    for ax in axes:
        ax.set_ylabel("Stress (kPa)")
        ax.set_xlim(0, None)  # Start x-axis at 0
        ax.set_ylim(bottom=0)  # Start y-axis at 0
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best')
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', alpha=0.2)
    
    # Add common x-label to the bottom subplot
    axes[-1].set_xlabel("Time (ms)")
    
    # Extract details from filename for the title
    base_name = os.path.basename(original_file)
    parts = base_name.replace('.csv', '').split('_')
    if len(parts) >= 5:
        afferent_type = parts[3]
        ramp = parts[4]
        scaling_factor = parts[5] if len(parts) > 5 else "1.0"
    else:
        afferent_type = "Unknown"
        ramp = "Unknown"
        scaling_factor = "Unknown"
    
    fig.suptitle(f"{afferent_type} Von Frey Stress Traces - Original vs Aligned to VF {reference_vf}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave some space for the title
    
    # Generate output filename if not provided
    if output_file is None:
        dir_name = os.path.dirname(original_file)
        base_name = os.path.basename(original_file)
        name_parts = os.path.splitext(base_name)
        output_file = os.path.join(dir_name, f"{name_parts[0]}_alignment_comparison.png")
    
    # Save the figure
    plt.savefig(output_file, dpi=150)
    logging.info(f"Comparison plot saved to {output_file}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Align stress traces to the 4.56 VF tip size.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file')
    parser.add_argument('--reference_vf', type=str, default='4.56', help='VF tip size to use as reference')
    parser.add_argument('--no_plot', action='store_true', help='Skip creating a comparison plot')
    
    args = parser.parse_args()
    
    # Align the stress traces
    aligned_file = align_to_reference_trace(
        args.input_file,
        args.output_file,
        args.reference_vf
    )
    
    if aligned_file and not args.no_plot:
        # Create a comparison plot
        plot_file = plot_comparison(
            args.input_file, 
            aligned_file, 
            reference_vf=args.reference_vf
        )
        print(f"Comparison plot saved to {plot_file}")
    
    print(f"Aligned traces saved to {aligned_file}")

if __name__ == "__main__":
    main() 
