# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:25:32 2016

@author: Administrator

Merat Updates 8/12/2022: the only way to get a good sinusoidal stress wave is to directly change it in the TipOneFive files under fem2.
"""
#outputs to "output2" in directory
import numpy as np
import pandas as pd
import os
import shutil
from multiprocessing import Pool
from collections import defaultdict, OrderedDict
from itertools import combinations
from lmfit import minimize, fit_report, Parameters
import matplotlib.pyplot as plt
import pickle
import datetime
import re
import copy

from stress_to_spike import (stress_to_fr_inst, spike_time_to_fr_roll,
                             spike_time_to_fr_inst)
from model_constants import (MC_GROUPS, FS, ANIMAL_LIST, STIM_NUM,
                             REF_ANIMAL, REF_STIM_LIST, WINDOW, REF_DISPL,
                             COLOR_LIST, CKO_ANIMAL_LIST)
from gen_function import get_interp_stress, stress_to_current
#from stress_gen_v1 import stress_generator

# Define parameters for fitting
# E.g., t3f1v23 means: 3 taus, fix tau1, vary tau2, tau3
lmpars_init_dict = {}
# Approach t2f12: use Adrienne's data
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True, min=0)
lmpars.add('k2', value=.3, vary=True, min=0)
lmpars.add('k3', value=.04, vary=True, min=0)
lmpars_init_dict['t2f12'] = lmpars

# Approach t2f12eqk23: use Adrienne's data, k3=k2
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True, min=0)
lmpars.add('k2', value=.3, vary=True, min=0)
lmpars.add('k3', expr='k2')
lmpars_init_dict['t2f12eqk23'] = lmpars

# Approach t2f12highk1: use Adrienne's data, k3!=k2, k1 = 10*n2
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True, min=0)
lmpars.add('k2', expr='k1 / 10')
lmpars.add('k3', value=.1, vary=True, min=0)
lmpars_init_dict['t2f12highk1'] = lmpars

# Approach t2v12: let tau1 and tau2 float
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=True, min=0, max=5000)
lmpars.add('tau2', value=200, vary=True, min=0, max=5000)
lmpars.add('tau3', value=np.inf, vary=False)
lmpars.add('k1', value=1, vary=True)
lmpars.add('k2', value=.3, vary=True)
lmpars.add('k3', value=.04, vary=True)
lmpars_init_dict['t2v12'] = lmpars

# Approach t3f1v23: fix tau1 and float tau2, 3
# Define lmpar
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False, min=0, max=5000)
lmpars.add('tau2', value=200, vary=True, min=0, max=5000)
lmpars.add('tau3', value=1832, vary=True, min=0, max=5000)
lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True)
lmpars.add('k2', value=.3, vary=True)
lmpars.add('k3', value=.04, vary=True)
lmpars.add('k4', value=.04, vary=True)
lmpars_init_dict['t3f1v23'] = lmpars

# Approach t3f123: add ultra-slow adapting constant and fix tau1, 2, 3
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=1832, vary=False)
lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True, min=0)
lmpars.add('k2', value=.5, vary=True, min=0)
lmpars.add('k3', value=.05, vary=True, min=0)
lmpars.add('k4', value=.05, vary=True, min=0)
lmpars_init_dict['t3f123'] = lmpars


# Approach t3f123eqk24: above but k2=k4
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=1832, vary=False)
lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True)
lmpars.add('k2', value=.3, vary=True)
lmpars.add('k3', value=.04, vary=True)
lmpars.add('k4', expr='k2')
lmpars_init_dict['t3f123eqk24'] = lmpars

# Approach t3f123highk1
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=1832, vary=False)
lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True, min=0)
lmpars.add('k2', expr='k1 / 20')
lmpars.add('k3', value=.04, vary=True, min=0)
lmpars.add('k4', value=.04, vary=True, min=0)
lmpars_init_dict['t3f123highk1'] = lmpars


# Approach t3f12v3
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=1832, vary=True) #drop it down to 0, you'll see a big difference in the spikes.

lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True, min=0)
lmpars.add('k2', value=.3, vary=True, min=0)
lmpars.add('k3', value=.04, vary=True, min=0)
lmpars.add('k4', value=.04, vary=True, min=0)
lmpars_init_dict['t3f12v3'] = lmpars


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


def get_mod_spike(lmpars, groups, time, stress):
    params = lmpars_to_params(lmpars)
    mod_spike_time, mod_fr_inst = stress_to_fr_inst(time, stress,
                                                    groups, **params)
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
        return os.path.join('data', 'rec', '%s_%s.csv' % (animal, datatype))
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

#I think this function is why the final output keeps resembling a ramp and hold.
def adjust_stress_ramp_time(time, stress, max_time_spike, stretch_coeff):
    """
    Stretch the ramp phase of the stress such that the ramp time matches the
    `max_time_target`. The hold phase of the stress is unchanged.

    Parameters
    ----------
    time : 1xN array
    stress : 1xN array
    max_time_spike : float
        When does the peak firing happens for the recording
    stretch_coeff : float
        The actual `max_time_target` is stretched by this coeff.

    Returns
    -------
    stress_new : 1XN array
    """
    # Clean the zeros in the beginning of the stress trace, if any
    start_idx = (stress == 0).nonzero()[0][-1]
    time = time[start_idx:] - time[start_idx]
    stress = stress[start_idx:]
    # Do the stretch
    max_idx = stress.argmax()
    max_time_original = time[max_idx]
    time_new = time.copy()
    max_time_target = max_time_spike * stretch_coeff
    time_new[:max_idx + 1] *= max_time_target / max_time_original
    time_new[max_idx + 1:] += max_time_target - max_time_original
    stress_new = np.interp(time, time_new, stress)
    return stress_new


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
    # Read model data, add the model data here:

    time, stress = get_interp_stress(static_displ)

    max_time = rec_dict['max_time_list'][stim]
#    stretch_coeff = 1 + 0.25 * static_displ / REF_DISPL
    stretch_coeff = 1 + 0.4 * static_displ / REF_DISPL
    #stress = adjust_stress_ramp_time(time, stress, max_time, stretch_coeff) #Yup, this line was the culprit. Now need to figure out how to do high stim and low stim
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
    result = minimize(get_single_residual,
                      lmpars, kws=fit_data_dict, epsfcn=1e-4)
    return result


def fit_single_rec_mp(args):
    return fit_single_rec(*args)


def plot_single_fit(lmpars_fit, groups, time, stress,
                    rec_spike_time, plot_kws={}, roll=True,
                    plot_rec=True, plot_mod=True,
                    fig=None, axs=None, save_data=False, fname=None,
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
    if plot_mod:
        mod_spike_time, mod_fr_inst = get_mod_spike(lmpars_fit, groups,
                                                    time, stress)
        print('mod_spike_time', mod_spike_time.shape, 'mod_fr', mod_fr_inst.shape)
        axs0.plot(mod_spike_time, mod_fr_inst * 1e3, '-', **plot_kws)
        axs0.set_xlim(0, 5000)
        axs0.set_xlabel('Time (msec)')
        axs0.set_ylabel('Instantaneous firing (Hz)')
        if save_data:
            np.savetxt('./data/%s_mod_spike_time.csv' % fname,
                       mod_spike_time, delimiter=',')
            np.savetxt('./data/%s_mod_fr_inst.csv' % fname,
                       mod_fr_inst * 1e3, delimiter=',')
    if plot_rec:
        axs1.plot(rec_spike_time, rec_fr * 1e3, '.', **plot_kws)
        axs1.set_xlim(0, 5000)
        axs1.set_xlabel('Time (msec)')
        axs1.set_ylabel('Instantaneous firing (Hz)')
        if save_data:
            np.savetxt('./data/%s_rec_spike_time.csv' % fname,
                       rec_spike_time, delimiter=',')
            np.savetxt('./data/%s_rec_fr_inst.csv' % fname,
                       rec_fr * 1e3, delimiter=',')
    fig.tight_layout()
    return fig, axs


def get_time_stamp():
    time_stamp = ''.join(re.findall(
        '\d+', str(datetime.datetime.now()))[:-1])
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


def get_lmpars_cko(lmpars, k_scale_dict):
    lmpars_cko = copy.deepcopy(lmpars)
    for k, scale in k_scale_dict.items():
        lmpars_cko[k].value *= scale
    return lmpars_cko


def get_params_paper(lmpars):
    """
    Generate the copy-pasteable table for paper writing
    """
    params_ser = pd.Series(lmpars.valuesdict())
    params_paper = pd.Series(dtype=np.float64)
    params_paper['tau1'] = params_ser['tau1']
    params_paper['tau2'] = params_ser['tau2']
    if 'tau4' in params_ser.keys():
        params_paper['tau3'] = params_ser['tau3']
    params_paper['knr'] = params_ser['k1']
    if 'tau4' in params_ser.keys():
        params_paper['kmc'] = params_ser['k2'] + params_ser['k4']
        params_paper['kmc1'] = params_ser['k2'] / params_paper['kmc']
        params_paper['kmc2'] = params_ser['k4'] / params_paper['kmc']
        params_paper['kusa'] = params_ser['k3']
    else:
        params_paper['kmc'] = params_ser['k2'] + params_ser['k3']
        params_paper['kmc1'] = params_ser['k2'] / params_paper['kmc']
        params_paper['kmc2'] = params_ser['k3'] / params_paper['kmc']
    return params_paper


def get_k_from_kmc(kmc, kmc1):
    k2 = kmc * kmc1
    k4 = kmc * (1. - kmc1)
    return k2, k4


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
            self.fit_ref()

    def load_rec_dicts(self):
        self.rec_dicts = {animal: load_rec(animal) for animal in ANIMAL_LIST}

    def get_data_dicts(self, animal, stim):
        data_dicts = get_data_dicts(stim, rec_dict=self.rec_dicts[animal])
        return data_dicts

    def load_data_dicts_dicts(self):
        self.data_dicts_dicts = {}
        for animal in ANIMAL_LIST:
            self.data_dicts_dicts[animal] = {}
            for stim in range(STIM_NUM):
                #print(STIM_NUM)
                self.data_dicts_dicts[animal][stim] = self.get_data_dicts(
                    animal, stim)

    def fit_ref(self, export=True):
        data_dicts_dict = self.data_dicts_dicts[REF_ANIMAL]
        # Prepare data for multiprocessing
        fit_mp_list = []
        for stim in REF_STIM_LIST:
            fit_mp_list.append([self.lmpars_init,
                                data_dicts_dict[stim]['fit_data_dict']])
        with Pool(5) as p:
            self.ref_result_list = p.map(fit_single_rec_mp, fit_mp_list)
        lmpar_list = [result.params for result in self.ref_result_list]
        self.ref_mean_lmpars = get_mean_lmpar(lmpar_list)
        # Plot the fit for multiple displacements
        if export:
            self.export_ref_fit()

    def export_ref_fit(self):
        pname = os.path.join('data', 'fit', self.label)
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
                **self.data_dicts_dicts[REF_ANIMAL][stim]['fit_data_dict'])
            plot_single_fit(
                self.ref_mean_lmpars, fig=fig, axs=axs[1], roll=roll,
                plot_kws={'color': color},
                **self.data_dicts_dicts[REF_ANIMAL][stim]['fit_data_dict'])
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
                    **self.data_dicts_dicts[animal_rec][stim]['fit_data_dict'])
            if animal_mod is not None:
                plot_single_fit(
                    lmpars_cko, fig=fig, axs=axs, roll=False,
                    plot_rec=False, plot_mod=True, save_data=save_data,
                    fname='%s_stim_%s' % (fname, {0: 'high', 2: 'low'}[stim]),
                    plot_kws={'color': color},
                    **self.data_dicts_dicts[animal_mod][stim]['fit_data_dict'])
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


if __name__ == '__main__':
    pass
    # %%
    fitApproach_dict = {}
    for approach, lmpars_init in lmpars_init_dict.items():
        lmpars_init = lmpars_init_dict[approach]
        fitApproach = FitApproach(lmpars_init, approach)
        fitApproach_dict[approach] = fitApproach
    # %% Figure 5
    fitApproach = fitApproach_dict['t3f123']
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    animal = 'Piezo2CONT'
    # Raw spikes
    for i, stim in enumerate(REF_STIM_LIST):
        # Spike timings of the model
        mod_spike_time, mod_fr_inst = get_mod_spike(
            fitApproach.ref_mean_lmpars,
            **fitApproach.data_dicts_dicts[animal][stim]['mod_data_dict'])
        plot_kws = dict(ymin=-.5 - stim, ymax=.5 - stim,
                        color=COLOR_LIST[stim], linewidth=.5)
        axs[0, 0].vlines(mod_spike_time, **plot_kws)
        axs[0, 0].axhline(-stim, color=COLOR_LIST[stim])
        axs[0, 0].set_ylim(-3.5, 1.5)
        axs[0, 0].set_ylabel('Spikes')
        axs[0, 0].get_yaxis().set_ticks([])
        # Add the bar on top
        #mod_peak_time = mod_spike_time[mod_fr_inst.argmax()]
        # axs[0, 0].plot([mod_peak_time, 5000], [2 - stim * .15, 2 - stim * .15],
        #                lw=4, c='.5', clip_on=False)
        # axs[0, 0].plot([0, mod_peak_time], [2 - stim * .15, 2 - stim * .15],
        #                lw=4, color='k', clip_on=False)
        axs[0, 0].set_xlim(0, 5000)
        axs[0, 0].set_xlabel('Time (msec)')
        # # Spike timings of the recording
        # rec_spike_time = fitApproach.rec_dicts[animal]['spike_time_list'][stim]
        # rec_fr_roll = fitApproach.rec_dicts[animal]['fr_roll_list'][stim]
        # axs[0, 1].vlines(rec_spike_time, **plot_kws)
        # axs[0, 1].axhline(-stim, color=COLOR_LIST[stim])
        # axs[0, 1].set_ylim(-3.5, 1.5)
        # axs[0, 1].set_ylabel('Spikes')
        # axs[0, 1].get_yaxis().set_ticks([])
        # Stress trace
        rec_spike_time = fitApproach.rec_dicts[animal]['spike_time_list'][stim]
        rec_fr_roll = fitApproach.rec_dicts[animal]['fr_roll_list'][stim]
        fine_time = fitApproach.data_dicts_dicts[animal][stim][
            'mod_data_dict']['time']
        #mean_stress = np.mean(fitApproach.data_dicts_dicts[animal][stim]['mod_data_dict']['stress'])
        #print(animal, ',', 'Stimulus: ', stim, 'Mean Stress: ', mean_stress, 'Pa')
        fine_stress = fitApproach.data_dicts_dicts[animal][stim][
            'mod_data_dict']['stress']
        axs[1, 0].plot(fine_time, fine_stress, color=COLOR_LIST[stim])
        #axs[1, 0].vlines(rec_spike_time, **plot_kws)
        #axs[1, 0].axhline(-stim, color=COLOR_LIST[stim])
        axs[1, 0].set_ylim(0, 100)
        axs[1, 0].set_ylabel('Stress')
        #axs[1, 0].get_yaxis().set_ticks([])

        # Add the bar on top
        rec_peak_time = rec_spike_time[rec_fr_roll.argmax()]
        # axs[0, 1].plot([rec_peak_time, 5000], [2 - stim * .15, 2 - stim * .15],
        #                lw=4, c='.5', clip_on=False)
        # axs[0, 1].plot([0, rec_peak_time], [2 - stim * .15, 2 - stim * .15],
        #                lw=4, color='k', clip_on=False)
        axs[0, 1].set_xlim(0, 5000)
        axs[0, 1].set_xlabel('Time (msec)')
    # Firing rates
    fitApproach.plot_cko_customized(
        {}, fig=fig, axs=axs[0, 1],
        animal_mod=animal, animal_rec=None)
    fitApproach.plot_cko_customized(
        {}, fig=fig, axs=axs[1, 1],
        animal_mod=None, animal_rec=animal)
    for axes in axs.ravel():
        axes.set_title('')
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.05, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.tight_layout()
    fig.subplots_adjust(top=.95)
    fig.savefig('./data/output2/fig5.png', dpi=300)
    fig.savefig('./data/output2/fig5.pdf', dpi=300)
    plt.close(fig)
    # %% Figure 6
    fitApproach = fitApproach_dict['t3f123']
    fig, axs = plt.subplots(3, 3, figsize=(7, 6))
    k_scale_dict_dict = {
        'Piezo2CONT': {},
        'Piezo2CKO': {'k4': 0},
        'Atoh1CKO': {'k2': 0, 'k4': 0}}
    for i, animal in enumerate(ANIMAL_LIST):
        lmpars_cko = get_lmpars_cko(fitApproach.ref_mean_lmpars,
                                    k_scale_dict_dict[animal])
        # Plot current
        for stim in REF_STIM_LIST:
            fine_time = fitApproach.data_dicts_dicts[animal][stim][
                'mod_data_dict']['time']
            mean_stress = np.mean(fitApproach.data_dicts_dicts[animal][stim]['mod_data_dict']['stress'])
            print(animal, ',', 'Stimulus: ', stim, 'Mean Stress: ',mean_stress, 'Pa')
            fine_stress = fitApproach.data_dicts_dicts[animal][stim][
                'mod_data_dict']['stress']
            params_dict = lmpars_to_params(lmpars_cko)
            single_current = stress_to_current(fine_time, fine_stress,
                                               **params_dict).sum(axis=1)
            axs[0, i].plot(fine_time, -single_current, color=COLOR_LIST[stim])
            axs[0, i].set_xlabel('Time (msec)')
            axs[0, i].set_ylabel('Current (pA)')
            axs[0, i].set_ylim(-20, 0)
            # Add the bar on top
            mod_peak_time = fine_time[single_current.argmax()]
            # axs[0, i].plot([mod_peak_time, 5000],
            #                [2 - stim * .5, 2 - stim * .5], '-',
            #                lw=4, c='.5', clip_on=False)
            # axs[0, i].plot([0, mod_peak_time],
            #                [2 - stim * .5, 2 - stim * .5], '-',
            #                lw=4, color='k', clip_on=False)
            axs[0, i].set_xlim(0, 5000)
        # Plot firing rate
        fitApproach.plot_cko_customized(
            k_scale_dict_dict[animal], fig=fig, axs=axs[1, i],
            animal_mod='Piezo2CONT', animal_rec=None)
        fitApproach.plot_cko_customized(
            k_scale_dict_dict[animal], fig=fig, axs=axs[2, i],
            animal_mod=None, animal_rec=animal)
        # Add the bar on top for rec
        for stim in REF_STIM_LIST:
            rec_spike_time = fitApproach.rec_dicts[animal]['spike_time_list'][
                stim]
            rec_fr_roll = fitApproach.rec_dicts[animal]['fr_roll_list'][stim]
            rec_peak_time = rec_spike_time[rec_fr_roll.argmax()]
            # axs[2, i].plot([rec_peak_time, 5000],
            #                [220 - stim * 5, 220 - stim * 5], '-',
            #                lw=4, c='.5', clip_on=False)
            # axs[2, i].plot([0, rec_peak_time],
            #                [220 - stim * 5, 220 - stim * 5], '-',
            #                lw=4, color='k', clip_on=False)
            axs[2, i].set_xlim(0, 5000)
    for axes in axs.ravel():
        axes.set_title('')
    fig.tight_layout()
    fig.tight_layout()
    fig.subplots_adjust(top=.95)
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.25, 1.05, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.savefig('./data/output2/fig6.png')
    fig.savefig('./data/output2/fig6.pdf')
    plt.close(fig)
    # %% Find a way for Piezo2 without totally kicking out the k4

    def ksa1_to_lmpars(ksa1, lmpars_old):
        lmpars_new = copy.deepcopy(lmpars_old)
        ksa = lmpars_old['k2'].value + lmpars_old['k4'].value
        lmpars_new['k2'].value = ksa * ksa1
        lmpars_new['k4'].value = ksa * (1 - ksa1)
        return lmpars_new

    def get_k_scale_dict(lmpars_old, lmpars_new):
        k_scale_dict = {
            key: lmpars_new[key].value / lmpars_old[key].value
            for key in ['k2', 'k4']}
        return k_scale_dict

    def ksa1_to_k_scale_dict(ksa1, lmpars_old):
        lmpars_new = ksa1_to_lmpars(ksa1, lmpars_old)
        k_scale_dict = get_k_scale_dict(lmpars_old, lmpars_new)
        return k_scale_dict

    for ksa1 in [.9, .95, .99, .999]:
        fitApproach.plot_cko_customized(
            ksa1_to_k_scale_dict(ksa1, fitApproach.ref_mean_lmpars),
            animal_rec='Atoh1CKO', animal_mod='Piezo2CONT')
    # %%
    k_scale_dict = {'k2': 1.15, 'k4': 0.01}
    lmpars = get_lmpars_cko(fitApproach.ref_mean_lmpars, k_scale_dict)
    params_ser = get_params_paper(lmpars)
    fitApproach.plot_cko_customized(
        k_scale_dict,
        animal_rec='Atoh1CKO', animal_mod='Piezo2CONT')
    # %% Figure 3
    animal = 'Piezo2CONT'
    stim = 0
    params_dict = lmpars_to_params(fitApproach.ref_mean_lmpars)
    fine_time = fitApproach.data_dicts_dicts[animal][stim][
        'mod_data_dict']['time']
    fine_stress = fitApproach.data_dicts_dicts[animal][stim][
        'mod_data_dict']['stress']
    # Control
    single_current = stress_to_current(fine_time, fine_stress,
                                       **params_dict)
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 6))
    current_label_list = ['RA', 'SA1', 'USA', 'SA2']
    for current, label in zip(single_current.T, current_label_list):
        axs[0].plot(fine_time, -current, label=label)
    axs[0].set_title('Piezo2 CONT')
    # Piezo2 CKO
    lmpars_cko = ksa1_to_lmpars(0.99, fitApproach.ref_mean_lmpars)
    params_dict_cko = lmpars_to_params(lmpars_cko)
    single_current_cko = stress_to_current(fine_time, fine_stress,
                                           **params_dict_cko)
    for current, label in zip(single_current_cko.T, current_label_list):
        axs[1].plot(fine_time, -current, label=label)
    axs[1].set_title('Piezo2 CKO')
    for axes in axs.ravel():
        axes.set_xlabel('Time (msec)')
        axes.set_ylabel('Current (pA)')
        axes.set_ylim(-8.5, 0.5)
        axes.legend(loc=4)
    fig.tight_layout()
    fig.savefig('./data/output2/current.png')
    plt.close(fig)
    # %% Current from neurite vs. mc
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 6))
    axs[0].plot(fine_time, -single_current.T[0] - single_current.T[2], '-k')
    axs[1].plot(fine_time, -single_current.T[1] - single_current.T[3], '-k')
    axs[2].plot(fine_time, -single_current.sum(axis=1), '-k')
    for axes in axs.ravel():
        axes.set_xlabel('Time (msec)')
        axes.set_ylabel('Current (pA)')
    axs[0].set_title('Neurite current')
    axs[1].set_title('Merkel cell current')
    axs[2].set_title('Generator current')
    fig.tight_layout()
    fig.savefig('./data/output2/current_components.png')
    plt.close(fig)
    # %% Effect of varying tau1
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    # Tau1
    tau1_list = [3, 8, 13]
    for tau1 in tau1_list:
        params_dict['tau_arr'][0] = tau1
        single_current = stress_to_current(fine_time, fine_stress,
                                           **params_dict)
        axs[0, 0].plot(fine_time, -single_current.T[0] - single_current.T[2])
    # Reset Tau1
    params_dict['tau_arr'][0] = 8
    # Tau2
    tau2_list = [50, 200, 350]
    for tau2 in tau2_list:
        params_dict['tau_arr'][1] = tau2
        single_current = stress_to_current(fine_time, fine_stress,
                                           **params_dict)
        axs[1, 0].plot(fine_time, -single_current.T[1] - single_current.T[3])
    plt.close(fig)
    # %% Generate the copy-pasteable table for paper writing
    params_paper_dict = {key: get_params_paper(value.ref_mean_lmpars)
                         for key, value in fitApproach_dict.items()}
    # %% Generate csv data for Greg plot #1
    fitApproach = fitApproach_dict['t3f12v3']
    animal = 'Piezo2CONT'
    stim = 2
    params_dict = lmpars_to_params(fitApproach.ref_mean_lmpars)
    fine_time = fitApproach.data_dicts_dicts[animal][stim][
        'mod_data_dict']['time']
    fine_stress = fitApproach.data_dicts_dicts[animal][stim][
        'mod_data_dict']['stress']
    # Control
    single_current = stress_to_current(fine_time, fine_stress,
                                       **params_dict)
    csv_dict = {}
    csv_dict['sa_current'] = single_current.T[1] + single_current.T[3]
    csv_dict['ra_current'] = single_current.T[0]
    csv_dict['time'] = np.arange(single_current.shape[0]) / 10
    csv_dict['usa_current'] = single_current.T[2]
    csv_dict['total_current'] = single_current.sum(axis=1)
    csv_dict['cluster_current'] = 8 * csv_dict['total_current']
    csv_dict['spike_time'], fr_inst = get_mod_spike(
        fitApproach.ref_mean_lmpars,
        **fitApproach.data_dicts_dicts[animal][stim]['mod_data_dict'])
    for key, item in csv_dict.items():
        np.savetxt('./data/GregCSVs2/fig1/%s.csv' % key, item, delimiter=',')
    # %% Parameter sensitivity figure
    # Get the k2 values
    kmc1_list = [.70, .865401, .99]
    kmc = fitApproach.ref_mean_lmpars['k2'] + fitApproach.ref_mean_lmpars['k4']
    k2_list = []
    k4_list = []
    for kmc1 in kmc1_list:
        k2_list.append(get_k_from_kmc(kmc, kmc1)[0])
        k4_list.append(get_k_from_kmc(kmc, kmc1)[1])

    def plot_current_and_spike(key, val, fig, axs,
                               fine_time=fine_time, fine_stress=fine_stress,
                               fitApproach=fitApproach,
                               **plotkws):
        # Initialization
        lmpars = copy.deepcopy(fitApproach.ref_mean_lmpars)
        lmpars[key].value = val
        params_dict_new = lmpars_to_params(lmpars)
        # Formatting data
        if key == 'k2':
            lmpars['k4'].value = k4_list[k2_list.index(val)]
            params_dict_new = lmpars_to_params(lmpars)
            keyname = 'kmc1'
            params_paper = get_params_paper(lmpars)
            intval = int(params_paper[keyname] * 100)
        elif key == 'tau1':
            keyname = 'tau_nr'
            intval = int(val)
        elif key == 'tau2':
            keyname = 'tau_mc'
            intval = int(val)
        # Calculation
        single_current = stress_to_current(fine_time, fine_stress,
                                           **params_dict_new).sum(axis=1)
        spike_time, fr_inst = get_mod_spike(
            lmpars,
            **fitApproach.data_dicts_dicts[animal][stim]['mod_data_dict'])
        # Plotting
        axs[0].plot(fine_time, single_current, '-',
                    label='%s=%d' % (keyname, intval),
                    **plotkws)
        axs[1].plot(spike_time, fr_inst * 1e3, '.',
                    label='%s=%d' % (keyname, intval),
                    **plotkws)
        # Saving data
        np.savetxt('./data/GregCSVs2/ParamTuning/current_%s_%d.csv' %
                   (keyname, intval), single_current, delimiter=',')
        np.savetxt('./data/GregCSVs2/ParamTuning/time_%s_%d.csv' %
                   (keyname, intval), fine_time, delimiter=',')
        np.savetxt('./data/GregCSVs2/ParamTuning/fr_inst_%s_%d.csv' %
                   (keyname, intval), fr_inst, delimiter=',')
        np.savetxt('./data/GregCSVs2/ParamTuning/spike_time_%s_%d.csv' %
                   (keyname, intval), spike_time, delimiter=',')
        return fig, axs
    fig, axs = plt.subplots(3, 2, figsize=(7, 7))
    param_tuning_dict = OrderedDict({'tau1': [1, 8, 15],
                                     'tau2': [50, 200, 350],
                                     'k2': k2_list})
    for axs_row, (key, val_list) in zip(axs, param_tuning_dict.items()):
        for level, val in enumerate(val_list):
            plot_current_and_spike(key, val, fig, axs_row,
                                   **{'color': str(0.3 * level)})
    for axes in axs.ravel():
        axes.legend()
    fig.savefig('./data/GregCSVs2/ParamTuning/example.png', dpi=300)

    # %% Fig 2
    fitApproach = fitApproach_dict['t3f12v3']
    k_scale_dict_dict = {
        'Piezo2CONT': {},
        'Piezo2CKO': {'k4': 0},
        'Atoh1CKO': {'k2': 0, 'k4': 0}}
    for i, animal in enumerate(ANIMAL_LIST):
        # Plot firing rate
        fitApproach.plot_cko_customized(
            k_scale_dict_dict[animal],
            animal_mod='Piezo2CONT', animal_rec=None,
            save_data=True, fname=animal)
        fitApproach.plot_cko_customized(
            k_scale_dict_dict[animal],
            animal_mod=None, animal_rec=animal,
            save_data=True, fname=animal)
    for fname in os.listdir('./data'):
        if fname.endswith('.csv'):
            shutil.move('./data/%s' % fname, './data/GregCSVs2/fig2/%s' % fname)
    # %% RA only
    try:
        fitApproach.plot_cko_customized(
            {'k2': 0, 'k3': 0, 'k4': 0}, animal_mod='Piezo2CONT',
            animal_rec=None, save_data=True, fname='RaOnly')
    except ValueError:
        pass
    for fname in os.listdir('./data'):
        if fname.endswith('.csv'):
            shutil.move('./data/%s' % fname,
                        './data/GregCSVs2/RaOnly/%s' % fname)
    # %% Fig #3
    fitApproach = fitApproach_dict['t3f12v3']
    animal = 'Piezo2CONT'
    stim = 2
    lmpars = fitApproach.ref_mean_lmpars
    fine_time = fitApproach.data_dicts_dicts[animal][stim][
        'mod_data_dict']['time']
    fine_stress = fitApproach.data_dicts_dicts[animal][stim][
        'mod_data_dict']['stress']
    # Control
    single_current = stress_to_current(fine_time, fine_stress,
                                       **params_dict)
    k_scale_dict_dict = {
        'Piezo2CONT': {},
        'Piezo2CKO': {'k4': 0},
        'Atoh1CKO': {'k2': 0, 'k4': 0}}
    csv_dict = {}
    # Generate SA current
    csv_dict['Piezo2CONT_sa_current'] = single_current.T[1] + \
                                        single_current.T[3]
    csv_dict['Piezo2CKO_sa_current'] = single_current.T[1]
    # Generate cumulate currents
    for animal, k_scale_dict in k_scale_dict_dict.items():
        lmpars_cko = get_lmpars_cko(lmpars, k_scale_dict)
        params_dict_cko = lmpars_to_params(lmpars_cko)
        single_current = stress_to_current(fine_time, fine_stress,
                                           **params_dict_cko).sum(axis=1)
        csv_dict['%s_current' % animal] = single_current
    time = np.arange(csv_dict['Piezo2CONT_current'].size) * 1e-1
    for key, item in csv_dict.items():
        np.savetxt('./data/GregCSVs2/fig3/B/%s.csv' % key, item, delimiter=',')
    np.savetxt('./data/GregCSVs2/fig3/B/time.csv', time, delimiter=',')
    # Copy data from fig 2 in
    for fname in os.listdir('./data/GregCSVs2/fig2'):
        if 'mod' in fname:
            shutil.copy('./data/GregCSVs2/fig2/%s' % fname,
                        './data/GregCSVs2/fig3/A/%s' % fname)
    # Separate RA, SA and USA currents
    # %% Fig #4
    method_dict = {'nousa': 't2f12',
                   'usa': 't3f12v3',
                   'longsa': 't2v12'}
    k_scale_dict_dict = {
        'Piezo2CONT': {},
        'Piezo2CKO': {'k4': 0},
        'Atoh1CKO': {'k2': 0, 'k4': 0}}
    for method_name, approach in method_dict.items():
        fitApproach = fitApproach_dict[approach]
#        for i, animal in enumerate(ANIMAL_LIST):
        animal = 'Piezo2CONT'
        # Plot firing rate
        fitApproach.plot_cko_customized(
            k_scale_dict_dict[animal],
            animal_mod='Piezo2CONT', animal_rec=None,
            save_data=True, fname=method_name + '_' + animal)
    for fname in os.listdir('./data'):
        if fname.endswith('.csv'):
            shutil.move('./data/%s' % fname,
                        './data/GregCSVs2/fig4/A/%s' % fname)
    method_name = 'usa'
    k_scale_dict = k_scale_dict_dict['Piezo2CKO']
    fitApproach = fitApproach_dict['t3f12v3']
    fitApproach.plot_cko_customized(
        k_scale_dict, animal_mod='Piezo2CONT', animal_rec=None,
        save_data=True, fname=method_name + '_' + 'Piezo2CKO')
    method_name, k_scale_dict = ('longsa', {'k3': 0})
    fitApproach = fitApproach_dict['t2v12']
    fitApproach.plot_cko_customized(
        k_scale_dict, animal_mod='Piezo2CONT', animal_rec=None,
        save_data=True, fname=method_name + '_' + 'Piezo2CKO')
    for fname in os.listdir('./data'):
        if fname.endswith('.csv'):
            shutil.move('./data/%s' % fname,
                        './data/GregCSVs2/fig4/B/%s' % fname)
