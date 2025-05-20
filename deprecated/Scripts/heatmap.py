import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def heatmap():
    data = pd.read_csv ("generated_csv_files/curved_stim_[4]mm_5kPa_interpolated_aggregated_simulation_results.csv")
    data['x_position'] = data['position'].apply(lambda pos: float(pos.split('_')[0][1:]))
    data['y_position'] = data['position'].apply(lambda pos: float(pos.split('_')[1][1:]))

    # Sum the SA spikes for each position
    data['total_SA_spikes'] = data['SA_spikes']
    bar_chart_data = data.groupby(['x_position', 'y_position'])['total_SA_spikes'].sum().reset_index()

    # Plot a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(bar_chart_data)), bar_chart_data['total_SA_spikes'], color='skyblue')
    plt.title('Total SA Spikes at Different Positions')
    plt.xlabel('Position Index')
    plt.ylabel('Total SA Spikes')
    plt.show()

if __name__ == '__main__':
    heatmap()



