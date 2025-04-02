import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def interpolate_stress_data(input_file, output_file=None):
    """
    Interpolate stress data to whole millisecond values from 0-5000ms.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file with aligned stress traces.
    output_file : str, optional
        Path to save the interpolated data. If None, output filename is generated.
    """
    # Load the data
    logging.info(f"Loading data from {input_file}")
    data = pd.read_csv(input_file)
    
    # Get column names for time and stress traces
    time_col = data.columns[0]
    stress_cols = [col for col in data.columns if col.startswith('Stress_')]
    
    # Create a new dataframe with whole millisecond values
    new_time = np.arange(0, 5001)  # 0 to 5000 inclusive
    new_data = pd.DataFrame({time_col: new_time})
    
    # Interpolate each stress column
    for col in stress_cols:
        logging.info(f"Interpolating {col}")
        # Use scipy's interp1d for linear interpolation
        new_data[col] = np.interp(new_time, data[time_col], data[col])
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.basename(input_file)
        name_parts = os.path.splitext(base_name)
        output_file = os.path.join(os.path.dirname(input_file), 
                                  f"{name_parts[0]}_interpolated{name_parts[1]}")
    
    # Save the interpolated data
    logging.info(f"Saving interpolated data to {output_file}")
    new_data.to_csv(output_file, index=False)
    
    return new_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Interpolate stress data to whole millisecond values.')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the input CSV file with aligned stress traces')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the interpolated data')
    
    args = parser.parse_args()
    
    # Interpolate the data
    interpolated_data = interpolate_stress_data(args.input_file, args.output_file)
    
    # Print information about the result
    logging.info(f"Interpolation complete. Output shape: {interpolated_data.shape}")
    logging.info(f"Time range: {interpolated_data[interpolated_data.columns[0]].min()} to {interpolated_data[interpolated_data.columns[0]].max()} ms")
    
if __name__ == "__main__":
    main() 
