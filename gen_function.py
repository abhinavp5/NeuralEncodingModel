# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:34:37 2014

@author: Lindsay

This script is used to construct and test the generator function for a single
Merkel cell-neurite connection.
The input of the function is stress, and output is current.
"""

import numpy as np
import os
from scipy.interpolate import interp1d
from model_constants import LIF_RESOLUTION, FE_NUM, DURATION, tau_brush, k_brush
from scipy import signal

#Should try diharmonic and triharmonic stimuli
# %% Generate Fine stress
def get_single_fine_stress(fe_id):
    rough_time, rough_force, rough_displ, rough_stress,\
        rough_strain, rough_sener = get_single_rough_fea(fe_id)

    fine_time, fine_stress = interpolate_stress(rough_time, rough_stress)
    fine_displ = interpolate_disp(rough_time, rough_displ)
    #fine_time = np.arange(0, 5000, LIF_RESOLUTION)
    #print('time', fine_time.shape)

    #fine_stress = 45*np.array()
    #fine_stress*0
    #fine_stress = np.multiply(45000,np.ones((len(fine_time),)))
    #print('stress',fine_stress)
    #fine_stress = abs(fine_stress)*abs(np.sin(30*fine_time))
    #fine_stress = fine_stress + 10*abs(np.sin(0.01*fine_time))

    return fine_time, fine_displ, fine_stress

#fine_time = np.arange(0, rough_time[-1], LIF_RESOLUTION)
#fine_stress = fine_stress-np.sin(2*np.pi*30*fine_time)

def get_single_rough_fea(fe_id):
    fname = 'TipOneFive%02dDispl.csv' % fe_id
    pname = os.path.join('data','fem',fname)
    #pname = os.path.join('data', 'fem',fname)
    #fname = 'RaInd4%dDispl.csv' % fe_id
    #pname = os.path.join('data', 'fem', 'RaInd4', fname)
    time, force, displ, stress, strain, sener = np.genfromtxt(
        pname, delimiter=',').T
    time *= 1e3  # sec to msec
    stress *= 1e-3  # Pa to kPa
    displ *= 1e3  # m to mm
    force *= 1e3  # N to mN
    sener *= 1e-3  # J m3 to kJ m3
    return time, force, displ, stress, strain, sener


def get_interp_stress(static_displ):
    """
    Get interpolated stress from FE model. Will do linear extrapolation.

    Parameters
    ----------
    static_displ : float
        The steady-state displ to scale the stress.

    Returns
    -------
    time : 1xN array
        Time array corresponding with the stress.
    stress : 1xN array
        Stress array.
    """
    time, static_displ_arr, stress_table = get_stress_table()
    if static_displ <= static_displ_arr[-1]:
        upper_index = (static_displ <= static_displ_arr).nonzero()[0][0]
        upper_displ = static_displ_arr[upper_index]
        lower_index = upper_index - 1
        lower_displ = static_displ_arr[lower_index]
        lower_ratio = (static_displ - lower_displ) / (upper_displ -
                                                      lower_displ)
        upper_ratio = 1 - lower_ratio
        stress = stress_table.T[lower_index] * lower_ratio +\
            stress_table.T[upper_index] * upper_ratio
    else:
        stress = stress_table.T[-1] + (static_displ - static_displ_arr[-1]) / (
            static_displ_arr[-1] - static_displ_arr[-2]) * (
            stress_table.T[-1] - stress_table.T[-2])
    return time, stress


def get_stress_table(fe_num=FE_NUM):
    """
    Parameters
    ----------
    fe_num : int
        Total number of fe runs.

    Returns
    -------
    time : 1xN array
        Time points.

    static_displ_arr : 1xM array
        A list for static displacements.

    stress_table : NxM array
        A table with columns as stress traces for each displacement.
    """
    time_list, displ_list, stress_list = [], [], []
    for fe_id in range(fe_num):
        time, displ, stress = get_single_fine_stress(fe_id)
        time_list.append(time)
        displ_list.append(displ)
        stress_list.append(stress)
    size_min = np.min([time.size for time in time_list])
    stress_table = np.column_stack(
        [np.zeros(size_min)] + [stress[:size_min] for stress in stress_list])
    static_displ_arr = np.r_[0, [displ[-1] for displ in displ_list]]
    return time[:size_min], static_displ_arr, stress_table


def get_interp_force(static_displ):
    """
    Get interpolated force from FE model. Will do linear extrapolation.

    Parameters
    ----------
    static_displ : float
        The steady-state displ to scale the force.

    Returns
    -------
    time : 1xN array
        Time array corresponding with the force.
    force : 1xN array
        Force array.
    """
    time, static_displ_arr, force_table = get_force_table()
    if static_displ <= static_displ_arr[-1]:
        upper_index = (static_displ <= static_displ_arr).nonzero()[0][0]
        upper_displ = static_displ_arr[upper_index]
        lower_index = upper_index - 1
        lower_displ = static_displ_arr[lower_index]
        lower_ratio = (static_displ - lower_displ) / (upper_displ -
                                                      lower_displ)
        upper_ratio = 1 - lower_ratio
        force = force_table.T[lower_index] * lower_ratio +\
            force_table.T[upper_index] * upper_ratio
    else:

        force = force_table.T[-1] + (static_displ - static_displ_arr[-1]) / (
            static_displ_arr[-1] - static_displ_arr[-2]) * (
            force_table.T[-1] - force_table.T[-2])
    return time, force


def get_force_table(fe_num=FE_NUM):
    """
    Parameters
    ----------
    fe_num : int
        Total number of fe runs.

    Returns
    -------
    time : 1xN array
        Time points.

    static_displ_arr : 1xM array
        A list for static displacements.

    force_table : NxM array
        A table with columns as force traces for each displacement.
    """
    time_list, displ_list, force_list = [], [], []
    for fe_id in range(fe_num):
        time, displ, force = get_single_fine_force(fe_id)
        time_list.append(time)
        displ_list.append(displ)
        force_list.append(force)
    size_min = np.min([time.size for time in time_list])
    force_table = np.column_stack(
        [np.zeros(size_min)] + [force[:size_min] for force in force_list])
    static_displ_arr = np.r_[0, [displ[-1] for displ in displ_list]]
    return time[:size_min], static_displ_arr, force_table


def get_single_fine_force(fe_id):
    rough_time, rough_force, rough_displ, rough_stress,\
        rough_strain, rough_sener = get_single_rough_fea(fe_id)
    fine_time, fine_force = interpolate_stress(rough_time, rough_force)
    fine_time, fine_displ = interpolate_stress(rough_time, rough_displ)
    return fine_time, fine_displ, fine_force


def interpolate_stress(rough_time, rough_stress):
    """
    Generate fine stress from rough stress using Linear Spline.

    Parameters
    ----------
    rough_time : 1d-array
        Timecourse from input stress file.
    rough_stress : 1d-array
        Stress from input stress file.

    Returns
    -------
    output_time_stress : 2d-array
        Fine time and Fine stress from Linear Spline of rough time and stress.
    """
    fine_time = np.arange(0, rough_time[-1], LIF_RESOLUTION)
    fine_spline = interp1d(rough_time, rough_stress, kind='slinear')
    fine_stress = fine_spline(fine_time)
    #fine_stress = fine_stress*np.sin(100*fine_time)
    return fine_time, fine_stress

def interpolate_disp(rough_time,rough_stress):
    """
    Generate fine stress from rough stress using Linear Spline.

    Parameters
    ----------
    rough_time : 1d-array
        Timecourse from input stress file.
    rough_stress : 1d-array
        Stress from input stress file.

    Returns
    -------
    output_time_stress : 2d-array
        Fine time and Fine stress from Linear Spline of rough time and stress.
    """
    fine_time = np.arange(0, rough_time[-1], LIF_RESOLUTION)
    fine_spline = interp1d(rough_time, rough_stress, kind='slinear')
    fine_stress = fine_spline(fine_time)
    #fine_stress = fine_stress*np.sin(100*fine_time)
    return fine_stress



# %% Convert stress to current
def stress_to_current(fine_time, fine_stress, tau_arr, k_arr): #, afferent_type='default', lateral_velocity=None):
    """
    Generate current from the stress of a single Merkel cell.

    Parameters
    ----------
    fine_time : 1xM array
        Time array of the indentation process.
    fine_stress : 1xM array
        Stress from a single Merkel cell.
    tau_arr : 1xN array
        Decay time constant for different adaptation mechanisms:
            tau_0, tau_1., ...., tau_inf
    k_arr : 1xN array
        Peak/steady ratio for different adaptation mechanisms.

    Returns
    -------
    current_arr : MxN array
        Generator current array from the generator function;
        each column represent one component.
    """
    # if afferent_type == 'Atoh1CKO':
    #     # Compute the instantaneous current as the product of lateral velocity and sensitivity factor k
    #     inst_current = k_brush * lateral_velocity
    #     # Define the exponential decay function
    #     decay_func = np.exp(-fine_time / tau_brush)
    #     # Convolve the instantaneous current with the exponential decay function to get the transduction current
    #     current_arr = np.array(signal.fftconvolve(inst_current, decay_func, mode='full')[:fine_time.size])
    #
    # else:
    #     # The current is calculated according to the existing formula
    #     ds = np.r_[0, np.diff(fine_stress)]
    #     k_func_arr = k_arr * np.exp(np.divide(-fine_time[None].T, tau_arr))
    #     current_arr = np.column_stack(
    #         [signal.fftconvolve(k_func_col, ds, mode='full')[:fine_time.size]
    #          for k_func_col in k_func_arr.T])
    #     current_arr[current_arr < 0] = 0
    # return current_arr
    #Working version:
    first_derivative = np.diff(fine_stress)
    #first_derivative = first_derivative/np.max(first_derivative) #normalize

    second_derivative = np.diff(fine_stress, n=2)
    #second_derivative = second_derivative/np.max(second_derivative) #normalize

    ds = np.r_[0, 0.4*first_derivative[:-1] + 1*second_derivative] #Used 0.2 for Figure 2 results and 0.4 for Figure 3 results for second derivative
    #print('Max INST Stress: ', np.max(ds))
    k_func_arr = k_arr * np.exp(np.divide(-fine_time[None].T, tau_arr))
    current_arr = np.column_stack(
        [signal.fftconvolve(k_func_col, ds, mode='full')[:fine_time.size]
         for k_func_col in k_func_arr.T])
    current_arr[current_arr < 0] = abs(current_arr[current_arr < 0])
    return current_arr




    ##original
    ##print('stress', fine_stress.shape, len(fine_stress))
    ##print('fine time size', fine_time.size, 'shape', fine_time.shape)
    #ds = np.r_[0, np.diff(fine_stress)]
    ##print(type(ds), 'ds shape', ds.shape)
    #k_func_arr = k_arr * np.exp(np.divide(-fine_time[None].T, tau_arr))
    ##print('k func', type(k_func_arr), k_func_arr.shape)
    #current_arr = np.column_stack(
    #    [signal.fftconvolve(k_func_col, ds.T, mode='full')[:fine_time.size]
    #     for k_func_col in k_func_arr.T])
    ##print('Current', type(current_arr), current_arr.shape)
    #current_arr[current_arr < 0] = 0

    #Modified
    # print('stress',fine_stress.shape, len(fine_stress))
    # ds = np.r_[np.zeros((1,7)),np.diff(fine_stress,axis=0)]
    # print(type(ds),'ds shape', ds.shape)
    # k_func_arr = k_arr * np.exp(np.divide(-fine_time[None].T, tau_arr))
    # print('k func', type(k_func_arr),k_func_arr.shape)
    # print('fine time size', fine_time.size)
    # current_arr = np.column_stack(
    #     [signal.fftconvolve(k_func_col, ds.T, mode='full',axes=0)[:fine_time.size]#+np.sin(2*np.pi*0.01*fine_time)
    #      for k_func_col in k_func_arr.T])
    # #current_arr = np.array([signal.fftconvolve(k_func_col, ds, mode='full')[:fine_time.size]  for k_func_col in k_func_arr.T])
    #
    # print('Current', type(current_arr), current_arr.shape)
    # current_arr[current_arr < 0] = 0


    # #current_arr = current_arr*abs(np.sin(fine_time)).T this is wrong
    #return current_arr

#I've been using this script to test modulating current instead of stress

# %% Main function
if __name__ == '__main__':
    pass
