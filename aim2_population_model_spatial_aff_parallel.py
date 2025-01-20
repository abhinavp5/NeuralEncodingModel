import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import concurrent.futures
import pandas as pd
import os 
import csv
import pickle 
import datetime
import re
import logging
import copy
import shutil
import sys
import ast
import argparse
from multiprocessing import Pool
from collections import defaultdict, OrderedDict
from itertools import combinations
from scipy.interpolate import interp1d
from lmfit import minimize, fit_report, Parameters
from stress_to_spike import (stress_to_fr_inst, spike_time_to_fr_roll,spike_time_to_fr_inst, spike_time_to_trace)
from model_constants import (MC_GROUPS, FS, ANIMAL_LIST, STIM_NUM, REF_ANIMAL, REF_STIM_LIST, WINDOW, REF_DISPL,COLOR_LIST, CKO_ANIMAL_LIST, k_brush, tau_brush, DURATION, LifConstants)
from gen_function import get_interp_stress, stress_to_current
from popul_model import pop_model
from fit_model_alt import FitApproach
import sys #for taking command line arguments

logger = logging.getLogger(__name__)
tic = time.time()
#set random seed for reproducibility
np.random.seed(0)

#Global Variables
lmpars_init_dict = {}
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=1744.6, vary=False)
lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=.74, vary=False, min=0) #a constant
lmpars.add('k2', value=.2088, vary=False, min=0) #b constant
lmpars.add('k3', value=.07, vary=False, min=0) #c constant
lmpars.add('k4', value=.0312, vary=False, min=0)
lmpars_init_dict['t3f12v3final'] = lmpars


def calculate_stress_at_position(stress, stimulus_diameter, rf_area, xk, yk, x_stimulus, y_stimulus, stimulus_type ):
    """
    Calculate the stress at the position based on different attributes including type of stimulus
    """
    if stimulus_type =="blunt":
        rs = stimulus_diameter / 2  # Radius of stimulus, half of diameter
        r_rf = np.sqrt(rf_area / np.pi)  # Radius of the receptive field
        squared_distance = (xk - x_stimulus) ** 2 + (yk - y_stimulus) ** 2
        rs_squared = rs ** 2
        r_rf_squared = r_rf ** 2
        tolerance = 1e-2  # Tolerance for floating-point comparison in mm

        # Calculate the squared distance for the condition when the edge of the stimulus aligns with the RF center
        edge_condition_squared_distance = rs_squared

        if abs(squared_distance - edge_condition_squared_distance) <= tolerance:  # Highest stress at the edge of the stimulus
            stress = stress  # Maximum stress remains unchanged
        elif squared_distance < rs_squared:  # Inside the stimulus
    

            # Stress decreases linearly to 0.2 * stress as it goes towards the center
            stress = ((squared_distance/rs_squared) * 0.8 + 0.2) * stress
        elif squared_distance > rs_squared and squared_distance <= (rs + r_rf) ** 2: # Inside the influence zone

            stress = ((-0.8 * squared_distance + (rs + r_rf) ** 2 - 0.2 * rs_squared)/ ((rs + r_rf) ** 2 - rs_squared)) * stress # Decreasing after the middle

        else:
            stress = np.zeros_like(stress)  # Outside the influence zone

        return stress
    
    # curved stimulus version 2.0, updated:
    elif stimulus_type == "curved":
        rs = stimulus_diameter / 2  # Radius of stimulus, half of diameter
        r_rf = np.sqrt(rf_area / np.pi)  # Radius of the receptive field
        R_curvature = 1000 / 365  # Radius of curvature in mm
        squared_distance = (xk - x_stimulus) ** 2 + (yk - y_stimulus) ** 2
        rs_squared = rs ** 2
        r_rf_squared = r_rf ** 2
        tolerance = 1e-2  # Tolerance for floating-point comparison in mm

        if squared_distance <= rs_squared:  # Inside the stimulus
            # Stress decreases from the center to the edge
            # Maximum stress at center, 0.2 * stress at edge
            # The stress decreases more gradually with higher curvature
            stress = (1 - 0.8*(squared_distance / rs_squared)) * stress
        elif squared_distance > rs_squared and squared_distance <= (rs + r_rf) ** 2:  # Inside the influence zone
            # Stress decreases further outside the stimulus
            # Ensure it starts from 0.2 at the edge of the stimulus and decreases to 0
            stress = ((-0.2 * (squared_distance - rs_squared) / ((rs + r_rf) ** 2 - rs_squared)) + 0.2) * stress
        else:
            stress = np.zeros_like(stress)  # Outside the influence zone

        return stress

def generate_afferent_positions_with_sizes(tongue_size, density_ratio, n_afferents, rf_sizes):
    """
    generates the positinos of afferents and theire relative sizes,based on arguments provided
    """
    length, width = tongue_size
    sa_density, ra_density = density_ratio

    total_density = sa_density + ra_density
    sa_probability = sa_density / total_density
    ra_probability = ra_density / total_density

    positions_with_sizes = []
    used_positions = set()  # Keep track of used positions

    while len(positions_with_sizes) < n_afferents:
        x = np.random.uniform(0, length)
        y = np.random.uniform(0, width)
        position = (x, y)  # Create a tuple for the position

        # Check if the position has already been used
        if position not in used_positions:
            afferent_type = np.random.choice(['SA', 'RA'], p=[sa_probability, ra_probability]) #SA or RA
            rf_size = np.random.choice(rf_sizes[afferent_type])
            positions_with_sizes.append((afferent_type, x, y, rf_size))
            used_positions.add(position)  # Mark this position as used

    return positions_with_sizes

def get_log_sample_after_peak(spike_time, fr_roll, n_sample):
    maxidx = fr_roll.argmax()
    log_range = np.logspace(0, np.log10(spike_time.size - maxidx),
                            n_sample).astype(np.int) - 1
    sample_range = maxidx + log_range
    return spike_time[sample_range], fr_roll[sample_range]

def get_single_residual(lmpars, groups,
                        time, stress, rec_spike_time, rec_fr_roll,
                        **kwargs):
    sample_spike_time, sample_fr_roll = get_log_sample_after_peak(
        rec_spike_time, rec_fr_roll, 50)
    mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress)
    mod_fr_inst_interp = np.interp(sample_spike_time,
                                   mod_spike_time, mod_fr_inst)
    residual = mod_fr_inst_interp - sample_fr_roll
    print((residual**2).sum())
    return residual
"""
Given Parameters return IFFs from stress traces
"""
def get_mod_spike(lmpars, groups, time, stress, g =.4, h = 1):
    
    params = lmpars_to_params(lmpars)
    mod_spike_time, mod_fr_inst = stress_to_fr_inst(time, stress,
                                                    groups,g=g,h=h,**params)
    return (mod_spike_time, mod_fr_inst)

def lmpars_to_params(lmpars):
    lmpars_dict = lmpars.valuesdict()
    # Export parameters to separate dicts and use indices as keys
    separate_dict = {'tau': {}, 'k': {}}
    for var, val in lmpars_dict.items():
        for param in separate_dict.keys():
            if param in var:
                index = int(var.split(param)[-1])
                separate_dict[param][index] = val
    # Convert to final format of parameter dict
    params = {}
    for param, indexed_dict in separate_dict.items():
        params[param + '_arr'] = np.array(
            np.array([indexed_dict[index]
                      for index in sorted(indexed_dict.keys())]))
    return params

def load_rec(animal):
    def get_fname(animal, datatype):
        return os.path.join('data/rec', '%s_%s.csv' % (animal, datatype))
    fname_dict = {datatype: get_fname(animal, datatype)
                  for datatype in ['spike', 'displ']}
    displ_arr = np.genfromtxt(fname_dict['displ'], delimiter=',')
    static_displ_list = np.round(displ_arr[-1], 2).tolist()
    spike_arr = np.genfromtxt(fname_dict['spike'], delimiter=',')
    spike_time_list = [spike.nonzero()[0] / FS for spike in spike_arr.T]
    fr_inst_list = [spike_time_to_fr_inst(spike_time)
                    for spike_time in spike_time_list]
    fr_roll_list, max_time_list, max_fr_roll_list = [], [], []
    for spike_time in spike_time_list:
        fr_roll = spike_time_to_fr_roll(spike_time, WINDOW)
        fr_roll_list.append(fr_roll)
        max_time_list.append(spike_time[fr_roll.argmax()])
        max_fr_roll_list.append(fr_roll.max())
    rec_dict = {
        'static_displ_list': static_displ_list,
        'spike_time_list': spike_time_list,
        'fr_inst_list': fr_inst_list,
        'fr_roll_list': fr_roll_list,
        'max_time_list': max_time_list,
        'max_fr_roll_list': max_fr_roll_list}
    return rec_dict

'''Function where stress traces is uploaded'''
def get_data_dicts(stim, animal=None, rec_dict=None):
    if rec_dict is None:
        rec_dict = load_rec(animal)
    # Read recording data
    rec_fr_inst = rec_dict['fr_inst_list'][stim]
    rec_spike_time = rec_dict['spike_time_list'][stim]
    rec_fr_roll = rec_dict['fr_roll_list'][stim]
    static_displ = rec_dict['static_displ_list'][stim]
    rec_data_dict = {
        'rec_spike_time': rec_spike_time,
        'rec_fr_inst': rec_fr_inst,
        'rec_fr_roll': rec_fr_roll}
    # Read model data
    #time, stress = get_interp_stress(static_displ)'
    # data = pd.read_csv("data/random_file.csv")
    # data = pd.read_csv('data/updated_dense_interpolated_stress_trace_RA.csv') #merats original stress traces
    data = pd.read_csv('data/vf_unscaled/3.61_out.csv') # Vonfrey Fillament w/ 0.4g
    # data = pd.read_csv('data/vf_unscaled/4.08_out.csv') # Vonfrey Fillament w/ 1pyt .0g
    # data = pd.read_csv('data/vf_unscaled/4.17_out.csv') # Vonfrey Fillament w/ 1.4g
    # data = pd.read_csv('data/vf_unscaled/4.31_out.csv') # Vonfrey Fillament w/ 2.0g
    # data = pd.read_csv('data/vonfreystresstraces/4.56_out.csv') # Vonfrey Fillament w/ 4.0g

    time = data['Time (ms)'].values
    stress = 0.1 * data[data.columns[1]].values

    #stress = weight * stress
    #time, stress = get_interp_stress(0.600001)
    max_time = rec_dict['max_time_list'][stim]
#    stretch_coeff = 1 + 0.25 * static_displ / REF_DISPL
    stretch_coeff = 1 + 0.4 * static_displ / REF_DISPL
    #stress = adjust_stress_ramp_time(time, stress, max_time, stretch_coeff)
    mod_data_dict = {
        'groups': MC_GROUPS,
        'time': time,
        'stress': stress}
    fit_data_dict = dict(list(rec_data_dict.items()) +
                         list(mod_data_dict.items()))
    data_dicts = {
        'rec_data_dict': rec_data_dict,
        'mod_data_dict': mod_data_dict,
        'fit_data_dict': fit_data_dict}
    
    return data_dicts

def fit_single_rec(lmpars, fit_data_dict):
    #minimizing the function get_single_residual
    result = minimize(get_single_residual,
                      lmpars, kws=fit_data_dict, epsfcn=1e-4)
    return result

def fit_single_rec_mp(args):
    return fit_single_rec(*args)

def plot_single_fit(lmpars_fit, groups, time, stress,
                    rec_spike_time, plot_kws={}, roll=True,
                    plot_rec=False, plot_mod=True,
                    fig=None, axs=None, save_data=True, fname="single_residual_model",
                    **kwargs):
    if roll:
        rec_fr = kwargs['rec_fr_roll']
    else:
        rec_fr = kwargs['rec_fr_inst']
    
    if fig is None and axs is None:
        fig, axs = plt.subplots()
        axs0 = axs
        axs1 = axs
    elif isinstance(axs, np.ndarray):
        axs0 = axs[0]
        axs1 = axs[1]
    else:
        axs0 = axs
        axs1 = axs
    
    # Plot modeled data (spike time and firing rate)
    if plot_mod:
        mod_spike_time, mod_fr_inst = get_mod_spike(lmpars_fit, groups, time, stress)
        axs0.plot(mod_spike_time, mod_fr_inst * 1e3, '-', **plot_kws, label="Modeled Firing Rate")
        axs0.set_xlim(0, 5000)
        axs0.set_xlabel('Time (msec)')
        axs0.set_ylabel('Instantaneous firing (Hz)')
        
        if save_data:
            np.savetxt('generated_plots/%s_mod_spike_time.csv' % fname, mod_spike_time, delimiter=',')
            np.savetxt('generated_plots/%s_mod_fr_inst.csv' % fname, mod_fr_inst * 1e3, delimiter=',')
    
    # Plot recorded data (spike time and firing rate)
    if plot_rec:
        axs1.plot(rec_spike_time, rec_fr * 1e3, '.', **plot_kws, label="Recorded Firing Rate")
        axs1.set_xlim(0, 5000)
        axs1.set_xlabel('Time (msec)')
        axs1.set_ylabel('Instantaneous firing (Hz)')
        
        if save_data:
            np.savetxt('generated_plots/%s_rec_spike_time.csv' % fname, rec_spike_time, delimiter=',')
            np.savetxt('generated_plots/%s_rec_fr_inst.csv' % fname, rec_fr * 1e3, delimiter=',')
    
    # New: Plot stress vs time for reference
    axs0.plot(time, stress, '--', color='gray', label="Stress vs Time")
    
    axs0.legend()
    fig.tight_layout()
    
    return fig, axs

###returns precise timestamp up to seconds mark in the format of a string
def get_time_stamp():
    time_stamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now()))[:-1])
    return time_stamp

def get_mean_lmpar(lmpar_list):
    if isinstance(lmpar_list, Parameters):
        return lmpar_list
    lmpar_dict_list = [lmpar.valuesdict() for lmpar in lmpar_list]
    all_param_dict = defaultdict(list)
    for lmpar_dict in lmpar_dict_list:
        for key, value in lmpar_dict.items():
            all_param_dict[key].append(value)
    mean_lmpar = copy.deepcopy(lmpar_list[0])
    for key, value in all_param_dict.items():
        mean_lmpar[key].set(value=np.mean(value))
    return mean_lmpar


class FitApproach():
    """
    If the parameters are stored in `data/fit`, then will load from file;
    Otherwise, fitting will be performed.
    """
    def __init__(self, lmpars_init, label=None):
        self.lmpars_init = lmpars_init
        if label is None:
            self.label = get_time_stamp()
        else:
            self.label = label
        # Load data
        self.load_rec_dicts()
        self.load_data_dicts_dicts()
        self.get_ref_fit()

    def get_ref_fit(self):
        pname = os.path.join('data', 'fit', self.label)
        if os.path.exists(pname):
            with open(os.path.join(pname, 'ref_mean_lmpars.pkl'), 'rb') as f:
                self.ref_mean_lmpars = pickle.load(f)
            self.ref_result_list = []
            for fname in os.listdir(pname):
                if fname.startswith('ref_fit') and fname.endswith('.pkl'):
                    with open(os.path.join(pname, fname), 'rb') as f:
                        self.ref_result_list.append(pickle.load(f))
        else:
            logging.warning("PERFORMING FITTING, DATA WAS NOT FOUND")
            self.fit_ref()

    def load_rec_dicts(self):
        self.rec_dicts = {animal: load_rec(animal) for animal in ANIMAL_LIST}

    def get_data_dicts(self, animal, stim):
        data_dicts = get_data_dicts(stim,rec_dict=self.rec_dicts[animal])
        return data_dicts

    def load_data_dicts_dicts(self):
        self.data_dicts_dicts = {}
        for animal in ANIMAL_LIST:
            self.data_dicts_dicts[animal] = {}
            for stim in range(STIM_NUM):
                self.data_dicts_dicts[animal][stim] = self.get_data_dicts(
                    animal, stim)

    def fit_ref(self, export=True):
        data_dicts_dict = self.data_dicts_dicts[REF_ANIMAL]
        # Prepare data for multiprocessing
        fit_mp_list = []
        for stim in REF_STIM_LIST:
            fit_mp_list.append([self.lmpars_init,
                                data_dicts_dict[stim]['mod_data_dict']])
        with Pool(5) as p:
            self.ref_result_list = p.map(fit_single_rec_mp, fit_mp_list)
        lmpar_list = [result.params for result in self.ref_result_list]
        self.ref_mean_lmpars = get_mean_lmpar(lmpar_list)
        # Plot the fit for multiple displacements
        if export:
            self.export_ref_fit()

    def export_ref_fit(self):
        pname = os.path.join('data/fem','data', 'fit', self.label) #path hard coded
        os.mkdir(pname)
        for stim, result in zip(REF_STIM_LIST, self.ref_result_list):
            fname_report = 'ref_fit_%d.txt' % stim
            fname_pickle = 'ref_fit_%d.pkl' % stim
            with open(os.path.join(pname, fname_report), 'w') as f:
                f.write(fit_report(result))
            with open(os.path.join(pname, fname_pickle), 'wb') as f:
                pickle.dump(result, f)
        with open(os.path.join(pname, 'ref_mean_lmpars.pkl'), 'wb') as f:
            pickle.dump(self.ref_mean_lmpars, f)
        # Plot
        fig, axs = self.plot_ref_fit(roll=True)
        fig.savefig(os.path.join(pname, 'ref_fit_roll.png'), dpi=300)
        plt.close(fig)
        fig, axs = self.plot_ref_fit(roll=False)
        fig.savefig(os.path.join(pname, 'ref_fit_inst.png'), dpi=300)
        plt.close(fig)

    def plot_ref_fit(self, roll=True):
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 6))
        for stim, ref_result in zip(REF_STIM_LIST, self.ref_result_list):
            lmpars_fit = ref_result.params
            color = COLOR_LIST[stim]
            plot_single_fit(
                lmpars_fit, fig=fig, axs=axs[0], roll=roll,
                plot_kws={'color': color},
                **self.data_dicts_dicts[REF_ANIMAL][stim]['mod_data_dict'])
            plot_single_fit(
                self.ref_mean_lmpars, fig=fig, axs=axs[1], roll=roll,
                plot_kws={'color': color},
                **self.data_dicts_dicts[REF_ANIMAL][stim]['mod_data_dict'])
        axs[0].set_title('Individual fitting parameters')
        axs[1].set_title('Using the average fitting parameter')
        fig.tight_layout()
        return fig, axs

    def plot_cko_customized(self, k_scale_dict,
                            animal_rec=None, animal_mod=None,
                            fig=None, axs=None,
                            close_fig=False, save_fig=False, show_label=False,
                            save_data=False, fname=''):
        lmpars_cko = copy.deepcopy(self.ref_mean_lmpars)
        for k, scale in k_scale_dict.items():
            lmpars_cko[k].value *= scale
        label = str(k_scale_dict).translate({ord(c): None for c in '{}\': .,'})
        if fig is None and axs is None:
            fig, axs = plt.subplots()
            close_fig = True
            save_fig = True
        for stim in REF_STIM_LIST:
            color = COLOR_LIST[stim]
            if animal_rec is not None:
                plot_single_fit(
                    lmpars_cko, fig=fig, axs=axs, roll=False,
                    plot_rec=True, plot_mod=False,
                    plot_kws={'color': color}, save_data=save_data,
                    fname='%s_stim_%s' % (fname, {0: 'high', 2: 'low'}[stim]),
                    **self.data_dicts_dicts[animal_rec][stim]['mod_data_dict'])
            if animal_mod is not None:
                plot_single_fit(
                    lmpars_cko, fig=fig, axs=axs, roll=False,
                    plot_rec=False, plot_mod=True, save_data=save_data,
                    fname='%s_stim_%s' % (fname, {0: 'high', 2: 'low'}[stim]),
                    plot_kws={'color': color},
                    **self.data_dicts_dicts[animal_mod][stim]['mod_data_dict'])
        if show_label:
            axs.set_title('Method: %s Rec: %s Mod: %s' %
                          (label, animal_rec, animal_mod))
        axs.set_ylim(0, 200)
        if save_fig:
            fig.tight_layout()
            fig.savefig('./data/output2/method_%s_rec_%s_mod_%s.png' %
                        (label, animal_rec, animal_mod))
        if close_fig:
            plt.close(fig)
        return fig, axs

#################TRYING CLASS BASED DESIGN TO IMPROVE BUILDABILITY AND READIBILIty############################
class Stimulus:
    def __init__(self, type, diameter,x_stim,y_stim):
        self.type = type
        self.diameter = diameter
        self.x_stim = x_stim
        self.y_stim = y_stim

    def get_lmpars_cko(self,lmpars, k_scale_dict):
        lmpars_cko = copy.deepcopy(lmpars)
        for k, scale in k_scale_dict.items():
            lmpars_cko[k].value *= scale
        return lmpars_cko
        
    '''
    filtered_spike_time:represents the spike times of the afferents post-filtereing
        for RA
            -applies a mask that excludes spikes that occur outside the [550ms,900nms]

    mod_spike_time  = array of spikes generated from stress input and afferent model parameters

    mod_fr_inst = stores the IFFS corresponding to spikes in mod_spike_time

    ********
    # first_spike_time = filtered_spike_time[0] - 368 if len(filtered_spike_time) > 0 else None
        - this statement is used in the case for the merat interpolated stress file
        - the first spike where stress is non-0 in the interpolated stree file is at 369ms

    ********

    
    
    
    '''
    def simulate_response(self, afferent):
        weights = [1.0]
        spikes_per_afferent = []
        # Placeholder for detailed simulation logic
        # Adapt the existing simulation logic here, working with the provided afferent_info and stimulus position (x_pos, y_pos)
        # For example, calculating decayed stress, simulating spike times, etc.
        # Return relevant results, such as spike counts, features, or any other metrics of interest

        fitApproach_dict = {}
        for approach, lmpars_init in lmpars_init_dict.items():
            lmpars_init = lmpars_init_dict[approach]
            fitApproach = FitApproach(lmpars_init, approach)  # added weights here to class FitApproach
            fitApproach_dict[approach] = fitApproach
        # %% Figure 5
        fitApproach = fitApproach_dict['t3f12v3final']
        # fig, axs = plt.subplots(4, 2, figsize=(7, 5))

        # final dictionary
        k_scale_dict_dict = {
            'SA': {'tau1': 8, 'tau2': 200, 'tau3': 1744.6, 'k1': .74, 'k2': 2.75, 'k3': .07, 'k4': .13},
            # SA 'k1':0, 'k3':0
            # 'Piezo2CKO': {'tau1':8, 'tau2':200,'tau3':1,'k1':80,'k2':0,'k3':0.0001,'k4':0 }} #RA when plotting SA
            'RA': {'tau1': 2.5, 'tau2': 200, 'tau3': 1, 'k1': 35, 'k2': 0, 'k3': 0, 'k4': 0}}  # RA parameters

        # 'Atoh1CKO': {'k1':k_brush} } #Brush, an very high k1 value is needed to get spikes for high frequency vibration
        # Raw spikes
        # for j, animal in enumerate(ANIMAL_LIST):
        lmpars_cko = self.get_lmpars_cko(fitApproach.ref_mean_lmpars,
                                    k_scale_dict_dict[afferent.afferent_type])

        for stim in REF_STIM_LIST:
            # column_name = data.columns[1] #column name of the second column
            # Spike timings of the model
            params_dict = lmpars_to_params(lmpars_cko)

            fine_time = fitApproach.data_dicts_dicts[afferent.afferent_type][stim]['mod_data_dict']['time']
            fine_stress = fitApproach.data_dicts_dicts[afferent.afferent_type][stim]['mod_data_dict']['stress']
            decayed_stress = calculate_stress_at_position(fine_stress, stimulus_diameter = self.diameter, rf_area=afferent.rf_size, xk=afferent.x_pos, yk=afferent.y_pos,
                                                        x_stimulus=self.x_stim, y_stimulus=self.y_stim, stimulus_type = self.type)
            # print("Decayed Stress Max: ", np.max(decayed_stress))
            mod_spike_time, mod_fr_inst = stress_to_fr_inst(

                fine_time,  # fitApproach.data_dicts_dicts[animal][stim]['mod_data_dict']['time'],
                decayed_stress,  # fitApproach.data_dicts_dicts[animal][stim]['mod_data_dict']['stress'],
                fitApproach.data_dicts_dicts[afferent.afferent_type][stim]['mod_data_dict']['groups'],
                **params_dict)


            # Lindsay's version:
            # mod_spike_time, mod_fr_inst = get_mod_spike(lmpars_cko,
            #     #fitApproach.ref_mean_lmpars,
            #     **fitApproach.data_dicts_dicts[animal][stim]['mod_data_dict'])

            # #remove
            if afferent.afferent_type == 'RA':

                mask = (mod_spike_time < 550) | (mod_spike_time > 900)
                filtered_spike_time = mod_spike_time  # mod_spike_time[mask]

                filtered_fr_inst = mod_fr_inst  # [mask]
            else:
                filtered_spike_time = mod_spike_time

                filtered_fr_inst = mod_fr_inst

            if len(mod_spike_time) == 0:
                # mod_spike_time = np.zeros_like(fine_time)
                # mod_fr_inst = np.zeros_like(fine_time)
                (afferent.afferent_type, "No Spikes at location")

            else:
                print(afferent.afferent_type, " First Spike Time: ", mod_spike_time[0], ' ms; ', )
                #print("Mean Firing Rate: ", np.mean(filtered_fr_inst) * 1e3, ' Hz')
                # print(animal, "Last Spike Time: ", mod_spike_time[-1], 'ms')

            fine_time = fitApproach.data_dicts_dicts[afferent.afferent_type][stim][
                'mod_data_dict']['time']
            # fine_time = fitApproach.data_dicts_dicts[animal][stim][
            #     'mod_data_dict'][column_name]['time']

            # feats, spikes = pop_model(mod_spike_time,mod_fr_inst)
            # print("SPIKES: ", spikes)
            # print('features type', type(feats))
            # features_animal[afferent_type].append(feats)
            # spikes_animal[afferent_type].append(spikes)
            # print("SPIKES ANIMAL: ", spikes_animal)
            # features = np.array(features)
            # print(animal, features_animal[animal])

            fine_time = fitApproach.data_dicts_dicts[afferent.afferent_type][stim][
                'mod_data_dict']['time']
            # mean_stress = np.mean(fitApproach.data_dicts_dicts[animal][stim]['mod_data_dict']['stress'])
            # print(animal, ',', 'Stimulus: ', stim, 'Mean Stress: ', mean_stress, 'Pa')
            fine_stress = fitApproach.data_dicts_dicts[afferent.afferent_type][stim][
                'mod_data_dict']['stress']

            # Plot firing rates
            # remove any spikes in between the bursts for RA, this is only for plotting purposes

            # isi_inst = np.r_[np.inf, np.diff(filtered_spike_time)]
            # iff_inst = 1 / isi_inst
            # print("filtered fr inst: ",filtered_fr_inst)
            # print("filtered spike time: ",filtered_spike_time)

            if len(filtered_spike_time) >= 2:
                # Proceed with interpolation as there are sufficient data points
                interp_function = interp1d(filtered_spike_time, filtered_fr_inst * 1e3, kind='slinear',
                                        bounds_error=False, fill_value=0)
                mod_fr_inst_interpolated = interp_function(fine_time)
            else:
                # Handle the case where there are not enough points to interpolate
                mod_fr_inst_interpolated = np.zeros_like(fine_time)
                if len(filtered_spike_time) == 1:
                    # Optionally, handle a single spike differently, e.g., by setting a non-zero rate up to the spike time
                    mod_fr_inst_interpolated[:np.searchsorted(fine_time, filtered_spike_time[0])] = 0
            # if there are no spikes in a region of time, then the interpolated firing rate will be 0.
            # Identify regions where no spikes occur
            # Assuming mod_spike_time is sorted
            no_spike_regions = np.diff(
                mod_spike_time) > 100  # Replace SOME_THRESHOLD with a suitable time gap
            no_spike_indices = np.where(no_spike_regions)[0]

            # Set mod_fr_inst_interpolated to zero in regions with no spikes
            for idx in no_spike_indices:
                start_time = filtered_spike_time[idx]
                end_time = filtered_spike_time[idx + 1]
                zero_indices = np.where((fine_time > start_time) & (fine_time < end_time))
                mod_fr_inst_interpolated[zero_indices] = 0

        feats, spikes = pop_model(mod_spike_time, mod_fr_inst_interpolated)
        spikes_per_afferent.append(len(filtered_spike_time))
        total_spikes = sum(spikes_per_afferent)
        mean_firing_frequency = np.mean(mod_fr_inst_interpolated) if len(filtered_fr_inst) > 0 else 0
        print("Mean Firing Rate: ", mean_firing_frequency, ' Hz') if mean_firing_frequency > 0 else None
        peak_firing_frequency = np.max(mod_fr_inst_interpolated) if len(filtered_fr_inst) > 0 else 0
        print("Peak Firing Rate: ", peak_firing_frequency, ' Hz') if peak_firing_frequency > 0 else None
        # first_spike_time = filtered_spike_time[0] - 368 if len(filtered_spike_time) > 0 else None
        first_spike_time = filtered_spike_time[0] if len(filtered_spike_time) > 0 else None
        # Placeholder return value
        return {
            'afferent_type': afferent.afferent_type,
            'x_pos': afferent.x_pos,
            'y_pos': afferent.y_pos,
            'spikes': total_spikes,  # Example: replace with actual spike count
            'activated': len(filtered_spike_time) > 0,  # True if the afferent is activated
            'stress': np.mean(fine_stress),
            'mean_firing_frequency': mean_firing_frequency,
            'peak_firing_frequency': peak_firing_frequency,
            'first_spike_time': first_spike_time
            # Include other relevant information as needed
        }

class Afferent:
    def __init__(self, afferent_type, x_pos, y_pos, rf_size):
        self.afferent_type = afferent_type
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.rf_size = rf_size
    def calculate_stress(self,stimulus):
        ###add logic to calculate stress###
        ###9/17/24 this method needs to be refactored to include stress as the first argument)
        return max(calculate_stress_at_position(stimulus.diameter, self.rf_size, self.x_pos, self.y_pos, stimulus.x_pos, stimulus.y_pos, stimulus.type))

class SimulationConfig:

    def __init__(self, tongue_size, density_ratio, n_afferents, rf_sizes, stimulus_type=None,
                stimulus_diameter=None, x_stimulus=None, y_stimulus=None, stress=None):
            
        self.stimulus_type = stimulus_type
        self.tongue_size = tongue_size
        self.density_ratio = density_ratio
        self.n_afferents = n_afferents
        self.rf_sizes = rf_sizes
        self.stimulus_diameter = stimulus_diameter
        self.x_stimulus = x_stimulus
        self.y_stimulus = y_stimulus
        self.stress = stress

        
    def generate_afferent_position(self):
        ###should be same as afferenet_positions_with_sizes####
        return generate_afferent_positions_with_sizes(self.tongue_size,self.density_ratio,self.n_afferents,self.rf_sizes)
    
    def set_stress(self, stress):
        self.stress = stress
    
       
class Simulation:
    def __init__(self, config : SimulationConfig):
        self.config = config
        if config.stimulus_type is not None:
            self.stimuli = [Stimulus(config.stimulus_type, dia, x, y )for dia in config.stimulus_diameter for x in config.x_stimulus for y in config.y_stimulus]
        if config.stimulus_diameter is not None:
            self.results_by_diameter = {dia: [] for dia in config.stimulus_diameter}
        self.afferents = [Afferent(aff_type,x,y,rf) for aff_type,x,y,rf  in config.generate_afferent_position()]


    def run(self):
        max_workers = os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # in dictionary comphrehension the key is future and the diameter value is associated with it
            futures = {executor.submit(self.simulate_afferent_response, afferent,stimulus): stimulus.diameter for stimulus in self.stimuli for afferent in self.afferents}
            for future in concurrent.futures.as_completed(futures):
                dia = futures[future]
                self.results_by_diameter[dia].append(future.result())##result is output of self.simulate_afferent_responses

            # print(self.results_by_diameter[dia])
            mean_firing_frequency = [afferent["mean_firing_frequency"] for afferent in self.results_by_diameter[dia]]
            peak_firing_frequency = [afferent["peak_firing_frequency"] for afferent in self.results_by_diameter[dia]]
            first_spike_time = [afferent["first_spike_time"] for afferent in self.results_by_diameter[dia]]

            
            activated = [d for d in self.results_by_diameter[dia] if d.get("activated", True) ==True]
            

                                 
            stresses = [afferent['stress'] for afferent in self.results_by_diameter[dia]]
  
            return stresses[-1], mean_firing_frequency, peak_firing_frequency, first_spike_time
    

    def simulate_afferent_response(self,afferent, stimulus):
        return stimulus.simulate_response(afferent)
    
    def post_process(self):
        # Post-processing of results
        # Example: Aggregate spikes per stimulus position and afferent type
        # Aggregate and print results for each diameter
        for dia, results in self.results_by_diameter.items():
            total_spikes_per_position = {}
            activated_afferents_per_position = {}
            for result in results:
                position_key = f"x{result['x_pos']}_y{result['y_pos']}_d{dia}"  # Include diameter in key
                afferent_type = result['afferent_type']
                spikes = result['spikes']
                activated = result['activated']

                if position_key not in total_spikes_per_position:
                    total_spikes_per_position[position_key] = {}
                if afferent_type not in total_spikes_per_position[position_key]:
                    total_spikes_per_position[position_key][afferent_type] = 0
                total_spikes_per_position[position_key][afferent_type] += spikes
                if position_key not in activated_afferents_per_position:
                    activated_afferents_per_position[position_key] = {'SA': 0, 'RA': 0}
                activated_afferents_per_position[position_key][afferent_type] += activated

            # print(f"Results for Diameter {dia} mm:")
            # for position, spikes_info in total_spikes_per_position.items():
            #     print(f"{position}: SA Spikes = {spikes_info.get('SA', 0)}, RA Spikes = {spikes_info.get('RA', 0)}")
            # for position, activation_info in activated_afferents_per_position.items():
            #     print(f"{position}: SA Activated = {activation_info['SA']}, RA Activated = {activation_info['RA']}")
            return total_spikes_per_position,activated_afferents_per_position
    def aggregate_results(self,config: SimulationConfig):
        # Aggregate results for each position
        aggregated_results = []
        for dia, results in self.results_by_diameter.items():
            position_data = {}
            for result in results:
                position_key = f"x{result['x_pos']}_y{result['y_pos']}_d{dia}"
                if position_key not in position_data:
                    position_data[position_key] = {
                        'SA': {'spikes': 0, 'activated': 0, 'mean_firing_frequency': 0, 'peak_firing_frequency': 0, 'first_spike_time': 0},
                        'RA': {'spikes': 0, 'activated': 0, 'mean_firing_frequency': 0, 'peak_firing_frequency': 0, 'first_spike_time': 0}
                    }
                afferent_type = result['afferent_type']
                position_data[position_key][afferent_type]['spikes'] += result['spikes']
                position_data[position_key][afferent_type]['activated'] += result['activated']
                position_data[position_key][afferent_type]['mean_firing_frequency'] += result['mean_firing_frequency']
                position_data[position_key][afferent_type]['peak_firing_frequency'] += result['peak_firing_frequency']
                if result['first_spike_time'] is not None:
                    position_data[position_key][afferent_type]['first_spike_time'] += result['first_spike_time']

            for position, data in position_data.items():
                aggregated_results.append({
                    'position': position,
                    'SA_spikes': data['SA']['spikes'],
                    'RA_spikes': data['RA']['spikes'],
                    'SA_activated': data['SA']['activated'],
                    'RA_activated': data['RA']['activated'],
                    'SA_mean_firing_frequency': data['SA']['mean_firing_frequency'],
                    'RA_mean_firing_frequency': data['RA']['mean_firing_frequency'],
                    'SA_peak_firing_frequency': data['SA']['peak_firing_frequency'],
                    'RA_peak_firing_frequency': data['RA']['peak_firing_frequency'],
                    'SA_first_spike_time': data['SA']['first_spike_time'],
                    'RA_first_spike_time': data['RA']['first_spike_time']
                })
    
        with open(f'generated_csv_files/{config.stimulus_type}_stim_{config.stimulus_diameter}mm_{config.stress}kPa_aggregated_simulation_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['position', 'SA_spikes', 'RA_spikes', 'SA_activated', 'RA_activated', 'SA_mean_firing_frequency',
                    'RA_mean_firing_frequency', 'SA_peak_firing_frequency', 'RA_peak_firing_frequency',
                    'SA_first_spike_time', 'RA_first_spike_time']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in aggregated_results:
                writer.writerow(row)
        return aggregated_results
         


    def plot(self, config: SimulationConfig):
        # Define colors for the afferent types
        colors = {'SA': '#31a354', 'RA': '#3182bd'}


        plt.figure(figsize=(10, 5))

        # Plot the stimulus locations as circles
        for x_stim, y_stim, stim_diameter in zip(config.x_stimulus, config.y_stimulus, config.stimulus_diameter):
            plt.gca().add_patch(
                patches.Circle((x_stim, y_stim), stim_diameter / 2, edgecolor='black', facecolor='none', linewidth=1)
            )

        # Loop through each afferent and plot its position
        
        x_positions = []
        y_positions = []
        calculated_alphas = []
        for afferent in self.afferents:
            # Calculate the stress level for each afferent based on its position and the nearest stimulus
            alpha = max(
                calculate_stress_at_position(
                    stress= config.stress,
                    stimulus_diameter=config.stimulus_diameter[0],  # Assuming single diameter for simplicity
                    rf_area=afferent.rf_size,
                    xk=afferent.x_pos, 
                    yk=afferent.y_pos, 
                    x_stimulus=x_stim, 
                    y_stimulus=y_stim,
                    stimulus_type=config.stimulus_type
                ) 
                for x_stim, y_stim in zip(config.x_stimulus, config.y_stimulus)
            )

            # Clip alpha between 0 and 1 to represent opacity
            #adding the main values to lists to writ eto csv
            x_positions.append(afferent.x_pos)
            y_positions.append(afferent.y_pos)
            calculated_alphas.append(alpha)
            if alpha>0:
                print(f"STRESS at X {afferent.x_pos} and Y {afferent.y_pos} is {alpha}")
            alpha = max(0, min(alpha, 1))
            
            # Set face color with RGBA for transparency based on stress
            facecolor_rgba = matplotlib.colors.to_rgba(colors[afferent.afferent_type], alpha)
            plt.scatter(afferent.x_pos, afferent.y_pos, facecolor=facecolor_rgba, edgecolor=colors[afferent.afferent_type],
                        s=np.pi * (afferent.rf_size) ** 2, label=afferent.afferent_type)
            
            

        # alphas_and_positions = set()

        # alphas_and_positions.add({
        #     ['x_pos']: tuple(x_positions),
        #     ['y_pos'] :tuple(y_positions),
        #     ["stress values"] : tuple(calculated_alphas)
        # })
        # with open(f'generated_csv_files/calculated_alpha_values.csv', 'w', newline='') as csvfile:
        #     headernames = ['x_positions', 'y_positions', 'alphas']

        #     writer = csv.DictWriter(csvfile, fieldnames=headernames)
        #     writer.writeheader()
        #     for row in alphas_and_positions:
        #         writer.writerow(row)



        # Set plot limits based on tongue size
        plt.xlim(0, config.tongue_size[0])
        plt.ylim(0, config.tongue_size[1])

        # Labeling the plot
        plt.xlabel('Length (mm)')
        plt.ylabel('Width (mm)')
        plt.title(f'Distribution of Afferents on Tongue Surface (n = {config.n_afferents}), {config.stimulus_type} Stimulus')
        plt.gca().set_aspect('equal', adjustable='box')

        # Create a legend for SA/RA types and the stimulus
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['SA'], markersize=10, label='SA'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['RA'], markersize=10, label='RA'),
            patches.Patch(color='black', label='Stimulus')
        ], loc='upper right')

        # Save the plot to a file
        plt.tight_layout()
        plt.savefig(f'generated_plots/{config.stimulus_type}_stim_{config.stimulus_diameter[0]}mm_{config.stress}kPa_afferent_distribution.png')
        plt.show()
    
    def get_afferents(self):
        return self.afferents



def main():
    # Check if command line arguments are provided
    start = time.perf_counter()
    if len(sys.argv) > 1 and sys.argv[1] == "lateral":
        print("LATERAL SIMULATION IS RUNNING")
        stimulus_type = sys.argv[2]  # "curved" or "blunt"
        tongue_size = ast.literal_eval(sys.argv[3])  # "(x,y)"
        density_ratio = ast.literal_eval(sys.argv[4])  # "(x,y)", x+y == 1
        n_afferents = int(sys.argv[5])  # Number of afferents
        rf_sizes = {
            'SA': ast.literal_eval(sys.argv[6]),  # List of SA uniform distributions "[a,b,c]"
            'RA': ast.literal_eval(sys.argv[7])   # List of RA uniform distributions "[a,b,c]"
        }
        stimulus_diameter = int(sys.argv[8])  # Single stimulus diameter
        x_start = int(sys.argv[9])  # Starting x-coord value
        x_end = int(sys.argv[10])  # Ending x-coord value
        x_step = 1
        y_stimulus = int(sys.argv[11])  # Y position (single value)
        stress = int(sys.argv[12])  # Stress in pascals

        # Generate lists for x_stimulus, y_stimulus, and stimulus_diameter
        x_stimulus = np.arange(x_start,x_end,x_step).tolist()
        y_stimulus = [y_stimulus] * len(x_stimulus)
        stimulus_diameter = [stimulus_diameter] * len(x_stimulus)

        # Convert to the same format as in the else block
        x_stimulus = ast.literal_eval(str(x_stimulus))
        y_stimulus = ast.literal_eval(str(y_stimulus))
        stimulus_diameter = ast.literal_eval(str(stimulus_diameter))
    elif len(sys.argv) == 11:
        stimulus_type = sys.argv[1]  # "curved" or "blunt"
        tongue_size = ast.literal_eval(sys.argv[2])  # "(x,y)"
        density_ratio = ast.literal_eval(sys.argv[3])  # "(x,y)", x+y == 1
        n_afferents = int(sys.argv[4])  # Number of afferents
        rf_sizes = {
            'SA': ast.literal_eval(sys.argv[5]),  # List of SA uniform distributions "[a,b,c]"
            'RA': ast.literal_eval(sys.argv[6])   # List of RA uniform distributions "[a,b,c]"
        }
        stimulus_diameter = ast.literal_eval(sys.argv[7])  # List of stimulus diameters "[a,...,n]"
        x_stimulus = ast.literal_eval(sys.argv[8])  # X positions of stimulus "[a,...,n]"
        y_stimulus = ast.literal_eval(sys.argv[9])  # Y positions of stimulus "[a,...,n]"
        stress = sys.argv[10] #stress in pascals?
    else:
        start = time.perf_counter()
        # Default values
        stimulus_type = "curved"
        tongue_size = (50, 25)  # in mm
        density_ratio = (0.44, 0.56)  # Ratio of SA and RA afferents
        n_afferents = 1000
        rf_sizes = {
            'SA': [1, 10, 19.6],
            'RA': [1, 6.5, 12.5]
        }
        stimulus_diameter = [4]  # Multiple diameters to simulate
        x_stimulus = [10]#x position of stimulus
        y_stimulus = [5] # y position of stimulus
        stress = 5 #stress in pascals placed placed by the stimulus
    # Create a SimulationConfig object
    
    config = SimulationConfig(tongue_size, density_ratio, n_afferents, rf_sizes,stimulus_type,stimulus_diameter, x_stimulus, y_stimulus,stress)

    # Initialize the Simulation class with the config
    simulation = Simulation(config)

    # Run the simulation returns the stress for the respective dat we're using
    calculated_stress, mean_firing_frequency, peak_firing_frequency, first_spike_time = simulation.run()


    print("THE CALCULATED STRESS IS:", calculated_stress)
    #setting the stress of config to the calculated stress
    # config.set_stress(calculated_stress)

    # # Post-process results
    simulation.post_process()

    #aggregates & saves reusults
    aggregated_data = simulation.aggregate_results(config)

    #plotting
    simulation.plot(config)

    #printing runtime
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')



if __name__ == '__main__':
    pass



