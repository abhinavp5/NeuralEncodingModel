# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 15:09:59 2014

@author: Lindsay

This module contains functions for running the LIF model.
"""

import numpy as np
from model_constants import (MC_GROUPS, LIF_PARAMS, LifConstants, DURATION,
                             REFRACTORY_PERIOD)


# %% Calculate du/dt
def dudt(pot, current):
    """
    Calculate du/dt using:
    du/dt = current / cm - pot / (rm * cm)

    Parameters
    ----------
    pot : double
        Potential at one heminode at the timepoint.
    current : double
        Transduction current at the timepoint.

    Returns
    -------
    ut : double
        The value of du/dt.
    """
    cm = LIF_PARAMS[1]
    rm = LIF_PARAMS[2]
    ut = current / cm - pot / (rm * cm)
    return ut


# %% Runge-Kutta calculation
def runge_kutta(current, start_time):
    """
    Use Runge-Kutta 4 to calculate the differential equation of LIF
    model and return timepoint and potential at the spike.

    Parameters
    ----------
    current : nd-array [timepoints, group number]
        Groups of transduction current in the total time duration.
        Group number = size of MC_GROUPS
    start_time : double
        The starting time for voltage calculation in the whole time duration.
        Voltage is 0 at the starting time.

    Returns
    -------
    output : array of 2 doubles
        [0] = timepoint when voltage at one heminode exceeds the threshold.
        [1] = largest voltage among all heminodes at the time.
    """
    threshold = LIF_PARAMS[0]
    mc_size = MC_GROUPS.shape[0]
    h = LifConstants.LIF_RESOLUTION
    time_span = DURATION - start_time
    temp_time = 0.
    current_index = int(start_time / h)
    each_pot = np.zeros(mc_size)
    max_pot = 0.
    mid_current = 0.
    k1 = 0.
    k2 = 0.
    k3 = 0.
    k4 = 0.
    output = np.zeros(2)
    while temp_time <= time_span-h:
        if current_index + 1 >= len(current):
            break
        for i in range(mc_size):
            mid_current = 0.5 * (current[current_index, i] +
                                 current[current_index + 1 , i]) # current_index + 1
            k1 = dudt(each_pot[i], current[current_index, i])
            k2 = dudt(each_pot[i] + 0.5*h*k1, mid_current)
            k3 = dudt(each_pot[i] + 0.5*h*k2, mid_current)
            k4 = dudt(each_pot[i] + h*k3, current[current_index + 1 , i]) # current_index + 1
            each_pot[i] = each_pot[i] + (h * (k1 + 2*k2 + 2*k3 + k4) / 6)
        max_pot = np.max(each_pot)
        if max_pot > threshold:
            break
        temp_time = temp_time + h
        current_index = current_index + 1  # current_index + 1
    output[0] = start_time + temp_time
    output[1] = max_pot
    return output


# %% Generate output spikes from input currents
def get_spikes(current):
    """
     Generate output spikes from input currents.

    Parameters
    ----------
    current : nd-array
        Grouped transduction current in the total time duration.
        Each column represents current from one group of Merkel cells.

    Returns
    -------
    spike_time_group : 1d_list
        Timepoints where there is a spike.
    """
    threshold = LIF_PARAMS[0]
    ini_time = 0.0
    integration_start = 0.0
    timestamp_finalpot = np.zeros([2])
    trace_length = int(DURATION / LifConstants.LIF_RESOLUTION + 1)
    print('trace_length = ', trace_length)
    spike_time = []
    larger_time = 0.0
    while ini_time <= DURATION:
        integration_start = ini_time
        timestamp_finalpot = runge_kutta(current[0:trace_length, :],
                                         integration_start)
        larger_time = np.max([timestamp_finalpot[0],
                              ini_time + REFRACTORY_PERIOD])
        if timestamp_finalpot[1] > threshold:
            spike_time.append(larger_time)
        ini_time = LifConstants.LIF_RESOLUTION + larger_time

    print(f"Generated spike times (before return): {spike_time[:5]}")  # Debugging spike times


    return spike_time


# %% Generate a set of standard normal-distributed noise and get moving average
# Not used in this calculation. Just put it here in case of future calculation.
def get_moving_avg_noise(noise_size, window_size):
    """
    Generate a set of standard normal-distributed noise and get moving average.

    Parameters
    ----------
    noise_size : int
        the length of the set
    window_size : int
        the length of the averaging window

    Returns
    -------
    mov_avg : 1d-array
        output noise set of which the length = noise_size
    """
    noise = np.random.normal(0, 1, noise_size+window_size-1)
    mov_avg = np.convolve(noise, np.ones(window_size)/window_size,
                          mode='valid')
    return mov_avg


# %% main function: test of LIF model
if __name__ == '__main__':
    pass
