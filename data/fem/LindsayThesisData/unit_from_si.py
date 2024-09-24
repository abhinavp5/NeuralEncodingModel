import os
import numpy as np


pname = 'unit_si'


def scale_and_save(fname, scale):
    data = np.genfromtxt(os.path.join(pname, fname), delimiter=',')
    scaled_data = data * scale
    np.savetxt(fname, scaled_data, delimiter=',')


if __name__ == '__main__':

    for fname in os.listdir(pname):
        if '_displ' in fname:
            scale = 1e3  # to mm
        elif '_force' in fname:
            scale = 1e3  # to mN
        elif '_time' in fname:
            scale = 1e3  # to ms
        elif '_stress' in fname:
            scale = 1e-3  # to kPa
        scale_and_save(fname, scale)
