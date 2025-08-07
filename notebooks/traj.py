# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:10:58 2025

@author: MKoshkareva
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys 
import os
import networkx as nx
from collections import Counter, defaultdict
import EoN
import tqdm
import time
import glob

# define path to the project for convenoent relative import
sys.path.append(os.path.dirname(os.getcwd())) 
from source.model_output import SEIRModelOutput, SEIRParams
from source.SEIR_network import SEIRNetworkModel


def save_seir_df(res, pop, ntype, seed,
                 beta, gamma, delta,
                 init_inf_frac, init_rec_frac):
    seed_df = pd.DataFrame([res.S, res.E, 
                        res.I, res.R]).T
    seed_df.columns = ['S','E','I','R']
    # use "values", because "iloc" saves index info 
    # and messes with calculation
    beta_calc = - seed_df.S.diff().values[1:] / (
                            seed_df.S.values[:-1] * seed_df.I.values[:-1]
                            )
    # the last Beta value cannot be calculated: no S_{t+1}
    seed_df['Beta'] = [*beta_calc, 0] 
    seed_df.fillna(0, inplace=True)
    
    params = [beta, gamma, delta, init_inf_frac, init_rec_frac]
    params_str = '_'.join([str(i) for i in params])
    
    seed_dir = f'../sim_data/new_{ntype}_{pop}/'
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)
    seed_df.to_csv(seed_dir + f'p_{params_str}_seed_{seed}.csv', 
                   index=False)
    

def create_all(ntype = 'r', pop = 10**4):
    tmax = 250 # time in days for simulation
    columns = ['beta', 'gamma', 'delta', 
               'init_inf_frac', 'init_rec_frac']+ \
              [day_index for day_index in range(tmax)]
    dataset = pd.DataFrame(columns=columns)
    
    
    chosen_seed = np.random.RandomState(42)
    network_model = SEIRNetworkModel(pop, ntype, chosen_seed)

    gamma = 0.1 # rate: E -> I
    delta = 0.08 # recovery rate: I -> R
    # fraction of initially infected

    # fraction of initially recovered
    init_rec_frac = 0
    # transmission rate
    beta_arr = np.arange(0.01, 0.2, 0.01) #np.arange(0.04, 0.09, 0.01)
    init_inf_frac_arr = np.arange(0.004, 0.021, 0.0005) #np.arange(0.005, 0.011, 0.001)

    n_runs = 10 #50
    times = []
    for beta in tqdm.tqdm(beta_arr):
        for init_inf_frac in init_inf_frac_arr:
            for seed in range(n_runs):
                start_time = time.time()
                res = network_model.simulate(beta=beta, gamma=gamma, 
                                             delta=delta, 
                                             init_inf_frac=init_inf_frac, 
                                             init_rec_frac=init_rec_frac,
                                             tmax=tmax)
                end_time = time.time()
                times.append(end_time-start_time)
                sample = [beta, gamma, delta, 
                          init_inf_frac, init_rec_frac] + res.daily_incidence
                dataset.loc[len(dataset)] = sample

                save_seir_df(res, pop, ntype, seed,
                             beta, gamma, delta,
                             init_inf_frac, init_rec_frac)
    return times    


if __name__ == '__main__':


    ntype = 'ba'
    pop = 1000   
    res = create_all(ntype, pop)
    pd.DataFrame(res).to_csv(f'../sim_data/time_{ntype}_{pop}.csv')
    
    
