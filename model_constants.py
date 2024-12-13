# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 10:56:20 2014

@author: Lindsay

This module contains all the constants needed in the LIF Model calculation.
"""

import numpy as np
import os




# %% LIF constants

class LifConstants:
    LIF_RESOLUTION = .5

    @classmethod
    def set_resolution(cls,value):
        cls.LIF_RESOLUTION = value


# LIF_RESOLUTION = .5
#DURATION = 1000 # in milliseconds
# DURATION = 5000 # in msec ##commenting this out but this was Merats original Duration
DURATION = 5000

REFRACTORY_PERIOD = 1  # in msec
MC_GROUPS = np.array([8, 5, 3, 1]) 
# MC_GROUPS = np.array([8,8,8,8])
#MC_GROUPS = np.array([8, 8, 8, 8])
# %% LIF_PARAMS
#    threshold (mV)
#    membrane capacitance (cm, in pF)
#    membrane resistance (rm, in Gohm)
#LIF_PARAMS = np.array([30, 30, 1.667])
# LIF_PARAMS = np.array([100., 100., 5.])
LIF_PARAMS = np.array([30., 30., 5]) #Original Version: [voltage mV, capacitance pF, resistance Gohm]
k_brush = 100
tau_brush = 300

# %% Recording constants
FS = 16  # kHz
ANIMAL_LIST = ['SA', 'RA'] #[SA, RA]
#ANIMAL_LIST = ['Piezo2CONT', 'Atoh1CKO']
#ANIMAL_LIST = ['Piezo2CONT'] #SA
#ANIMAL_LIST = ['Piezo2CKO'] #RA
#ANIMAL_LIST = ['SA','RA']


#REF_ANIMAL = 'Piezo2CONT'
REF_ANIMAL = 'SA'


CKO_ANIMAL_LIST = ['Piezo2CKO', 'Atoh1CKO']
#CKO_ANIMAL_LIST = ['RA']


MAT_FNAME_DICT = {
    'Piezo2CONT': '2013-12-07-01Piezo2CONT_calibrated.mat',
    'Piezo2CKO': '2013-12-13-02Piezo2CKO_calibrated.mat',
    'Atoh1CKO': '2013-10-16-01Atoh1CKO_calibrated.mat'}

#
STIM_LIST_DICT = {
    'Piezo2CONT': [(101, 2), (101, 3), (101, 1)],
    'Piezo2CKO': [(201, 2), (201, 7), (201, 4)]}
#     'Atoh1CKO': [(101, 2), (101, 1), (101, 5)]}

# STIM_LIST_DICT = {#Original
#     'Piezo2CONT': [(101, 2), (101, 3), (101, 1)],
#     'Atoh1CKO': [(101, 2), (101, 1), (101, 5)]}

# STIM_LIST_DICT = {
#     'Piezo2CONT': [(101, 2), (101, 3), (101, 1)],
#     'Atoh1CKO': [(101, 2), (101, 3), (101, 1)]}

# STIM_LIST_DICT = {
#     'SA': [(101, 2), (101, 3), (101, 1)],
#     'RA': [(101, 2), (101, 1), (101, 5)]}

STIM_NUM = len(next(iter(STIM_LIST_DICT.values())))
#print(STIM_NUM)
REF_STIM = 0
#REF_STIM_LIST = [0, 2, 4, 6, 8, 10, 12]
#REF_STIM_LIST = [0] #high_stim
REF_STIM_LIST = [2] #low_stim
#REF_STIM_LIST = [0,2]
WINDOW = 5

fempath = os.path.normpath('data/fem') ##hardcoded file path
# %% FEM constants
fe_id_list = [int(fname[10:12])
              for fname in os.listdir(fempath) if fname.endswith('csv')]
#fe_id_list = [int(fname[6:7])
#              for fname in os.listdir('data/fem/RaInd4') if fname.endswith('csv')]
FE_NUM = np.max(fe_id_list) + 1
REF_DISPL = .6

# %% For plotting
COLOR_LIST = ['.5', 'r', 'k', 'b', 'c', 'm', 'y', 'r', 'g', 'b']

