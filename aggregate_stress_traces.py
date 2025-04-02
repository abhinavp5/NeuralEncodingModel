import pandas as pd
import numpy as np
import os
import logging
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate_stress_traces(afferent_type, ramp="out", scaling_factor=1.0):
    """
    Aggregate stress traces from different VF tip sizes into a single CSV file.
    
    Parameters:
    -----------
    afferent_type : str
        The type of afferent, e.g., "RA" or "SA".
    ramp : str, optional
        The ramp type, default is "out".
    scaling_factor : float, optional
        Scaling factor for stress values, default is 1.0.
    
    Returns:
    --------
    str
        Path to the saved CSV file.
    """
    # Define VF tip sizes
    vf_tip_sizes = [3.61, 4.08, 4.17, 4.31, 4.56]
    
    # Create a dictionary to store the original data
    data_dict = {}
    
    # Create directories if they don't exist
    os.makedirs("aggregated_data", exist_ok=True)
    
    # First, collect all data with their original time axes
    for vf in vf_tip_sizes:
        try:
            # Determine the file path based on VF tip size
            if vf == 4.56:
                file_path = f"data/P3/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv"
            else:
                file_path = f"data/P4/Realistic/{vf}/{vf}_radial_stress_corr_realistic.csv"
            
            # Read the CSV file
            data = pd.read_csv(file_path)
            logging.info(f"Successfully read data for VF {vf}")
            
            # Store the time and stress data
            data_dict[vf] = {
                'time': data['Time (ms)'].to_numpy(),
                'stress': scaling_factor * data[data.columns[1]].values
            }
            
        except Exception as e:
            logging.error(f"Error reading data for VF {vf}: {str(e)}")
    
    # Check if we have data
    if not data_dict:
        logging.error("No data was successfully loaded.")
        return None
    
    # Find the common time range
    min_times = [data['time'].min() for data in data_dict.values()]
    max_times = [data['time'].max() for data in data_dict.values()]
    
    common_min_time = max(min_times)
    common_max_time = min(max_times)
    
    logging.info(f"Common time range: {common_min_time} to {common_max_time} ms")
    
    # Create a common time grid
    num_points = 1000  # Adjust as needed for desired resolution
    common_time = np.linspace(common_min_time, common_max_time, num_points)
    
    # Create a new dictionary for the interpolated data
    stress_traces = {'Time (ms)': common_time}
    
    # Interpolate each stress trace to the common time grid
    for vf in vf_tip_sizes:
        if vf in data_dict:
            # Create an interpolation function
            interp_func = interp1d(
                data_dict[vf]['time'], 
                data_dict[vf]['stress'], 
                bounds_error=False,  # Allow extrapolation
                fill_value=0.0        # Fill with zeros outside the original domain
            )
            
            # Interpolate to the common time grid
            interpolated_stress = interp_func(common_time)
            
            # Store the interpolated data
            stress_traces[f"Stress_{vf}"] = interpolated_stress
        else:
            # If data for this VF tip size was not loaded, add zeros
            stress_traces[f"Stress_{vf}"] = np.zeros_like(common_time)
    
    # Create a pandas DataFrame from the dictionary
    stress_df = pd.DataFrame(stress_traces)
    
    # Save the DataFrame to a CSV file
    output_file = f"aggregated_data/aggregated_stress_traces_{afferent_type}_{ramp}_{scaling_factor}.csv"
    stress_df.to_csv(output_file, index=False)
    logging.info(f"Aggregated stress traces saved to {output_file}")
    
    return output_file

def plot_stress_traces(csv_file, title=None):
    """
    Create a plot of the aggregated stress traces.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file with aggregated stress traces.
    title : str, optional
        Custom title for the plot. If None, a default title is generated.
    
    Returns:
    --------
    str
        Path to the saved plot file.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Extract the time column
    time = data['Time (ms)']
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Colors for the different VF tip sizes
    colors = ['#440154', '#3b528b', '#21908c', '#5dc963', '#fde725']
    
    # Plot each stress trace
    for i, col in enumerate(data.columns):
        if col != 'Time (ms)':
            # Extract the VF tip size from the column name
            vf_size = col.split('_')[1]
            plt.plot(time, data[col], label=f'VF {vf_size}', color=colors[i-1])
    
    # Add labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('Stress (kPa)')
    
    if title is None:
        # Extract the information from the CSV file name
        filename = os.path.basename(csv_file)
        parts = filename.replace('.csv', '').split('_')
        afferent_type = parts[3]
        ramp = parts[4]
        scaling_factor = parts[5]
        title = f"{afferent_type} Von Frey Stress Traces (Ramp: {ramp}, Scaling: {scaling_factor})"
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Configure axes
    plt.xlim(left=0)  # Set x-axis to start at 0
    plt.ylim(bottom=0)  # Set y-axis to start at 0
    
    # Add minor ticks and grid
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Save the figure
    plot_file = csv_file.replace('.csv', '.png')
    plt.savefig(plot_file)
    logging.info(f"Plot saved to {plot_file}")
    
    # Close the figure to free memory
    plt.close()
    
    return plot_file

def main():
    # Parse command line arguments if needed
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate stress traces from different VF tip sizes.')
    parser.add_argument('--afferent_type', type=str, default="SA", help='Afferent type (RA or SA)')
    parser.add_argument('--ramp', type=str, default="out", help='Ramp type')
    parser.add_argument('--scaling_factor', type=float, default=1.0, help='Scaling factor for stress values')
    parser.add_argument('--no_plot', action='store_true', help='Skip plotting the aggregated data')
    
    args = parser.parse_args()
    
    # Run the aggregation function
    output_file = aggregate_stress_traces(
        afferent_type=args.afferent_type,
        ramp=args.ramp,
        scaling_factor=args.scaling_factor
    )
    
    if output_file:
        print(f"Aggregated stress traces saved to {output_file}")
        
        # Generate a plot if not disabled
        if not args.no_plot:
            plot_file = plot_stress_traces(output_file)
            print(f"Plot saved to {plot_file}")
    else:
        print("Failed to aggregate stress traces.")

if __name__ == "__main__":
    main() 
