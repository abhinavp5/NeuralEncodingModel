from gen_function import (interpolate_stress, get_interp_stress, stress_to_current)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("data/vonfreystresstraces/4.56_out.csv")
    time = data['Time (ms)'].values
    rough_stress = 0.01 * data[data.columns[1]].values
    fine_time, fine_stress = interpolate_stress(rough_time = time, rough_stress= rough_stress)

    print("Rough", rough_stress)
    print("Fine", fine_stress)
    ##There is a change between fine stress and rought stress
    
 

if __name__ == '__main__':
  main()
