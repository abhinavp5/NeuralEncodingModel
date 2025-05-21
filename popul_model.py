#This script will create a population model from receptors and calculate sum of firing rates, peak firing rates, and number of afferents used after spikes
# spikes are generated from lif model.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import time
from lif_model import get_spikes
from model_constants import (LifConstants, DURATION, MC_GROUPS)
from gen_function import stress_to_current


# def pop_model(spike_time, fr_inst):
#     """
#     Create population model from spike time and instantaneous firing rate.
#
#     Parameters
#     ----------
#     spike_time : 1d-array
#         The time points of all the spikes in msec.
#     fr_inst : 1d-array
#         The instantaneous firing rate of the population.
#
#     Returns
#     -------
#     fr_sum : float
#         Sum of firing rate.
#     fr_peak : float
#         Peak firing rate.
#     n_aff : float
#         Number of afferents used.
#     """
#     #fr_sum = fr_inst.sum()
#     #fr_peak = fr_inst.max()
#     features = [np.sum(fr_inst*1e3),len(spike_time), np.sum(np.max(fr_inst*1e3))]
#     spikes = len(spike_time)
#     # np.min(fr_inst*1e3), np.percentile(fr_inst*1e3, 25), np.percentile(fr_inst*1e3, 75), np.percentile(fr_inst*1e3, 90), np.percentile(fr_inst*1e3, 95), np.percentile(fr_inst*1e3, 99)]
#
#     #n_aff = len(spike_time)
#     return features, spikes

# def pop_model(spike_time, fr_inst):
#     """
#     Create population model from spike time and instantaneous firing rate.
#
#     Parameters
#     ----------
#     spike_time : 1d-array
#         The time points of all the spikes in msec.
#     fr_inst : 1d-array
#         The instantaneous firing rate of the population.
#
#     Returns
#     -------
#     features : list
#         List of various metrics calculated from the firing rates and spike times.
#     n_spikes : int
#         Total number of spikes.
#     """
#     if len(spike_time) == 0 or len(fr_inst) == 0:
#         # If no spikes or no firing rates, return None for all values
#         features = [None]*6
#         n_spikes = 0
#     else:
#         fr_sum = np.sum(fr_inst*1e3)
#         fr_avg = np.mean(fr_inst*1e3)
#         fr_peak = np.max(fr_inst*1e3)
#         first_spike_time = np.min(spike_time)
#         last_spike_time = np.max(spike_time)
#         n_spikes = len(spike_time)
#
#         features = [fr_sum, fr_avg, fr_peak, first_spike_time, last_spike_time, n_spikes]
#
#     return features, n_spikes


def pop_model(spike_time, fr_inst):
    """
    Create population model from spike time and instantaneous firing rate.

    Parameters
    ----------
    spike_time : 1d-array
        The time points of all the spikes in msec.
    fr_inst : 1d-array
        The instantaneous firing rate of the population.

    Returns
    -------
    features : dict
        Dictionary of various metrics calculated from the firing rates and spike times.
    n_spikes : int
        Total number of spikes.
    """
    if len(spike_time) == 0 or len(fr_inst) == 0:
        # If no spikes or no firing rates, return None for all values
        features = {
            'Sum Firing Rate': None,
            'Average Firing Rate': None,
            'Peak Firing Rate': None,
            'Sum First Spike Time': None,
            'Sum Last Spike Time': None,
            'Number of Spikes': None,
        }
        n_spikes = 0
    else:
        fr_sum = np.sum(fr_inst*1e3)
        fr_avg = np.mean(fr_inst*1e3)
        fr_peak = np.max(fr_inst*1e3)
        first_spike_time = spike_time[0]
        last_spike_time = spike_time[-1]
        n_spikes = len(spike_time)

        features = {
            'Sum Firing Rate': fr_sum,
            'Average Firing Rate': fr_avg,
            'Peak Firing Rate': fr_peak,
            'Sum First Spike Time': first_spike_time,
            'Sum Last Spike Time': last_spike_time,
            'Number of Spikes': n_spikes,
        }

    return features, n_spikes
