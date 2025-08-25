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
    tmax = 150 # time in days for simulation
    columns = ['beta', 'gamma', 'delta', 
               'init_inf_frac', 'alpha']+ \
              [day_index for day_index in range(tmax)]
    dataset = pd.DataFrame(columns=columns)
    
    
    chosen_seed = np.random.RandomState(42)
    network_model = SEIRNetworkModel(pop, ntype, chosen_seed)

    #sigma = 0.1 # rate: E -> I
    #gamma = 0.08 # recovery rate: I -> R
    
    gamma = 0.3 # latent period rate
    delta = 0.2 # recovery rate
    # fraction of initially infected

    # fraction of initially recovered
    init_inf_frac = 1e-4 # fraction of initially infected
    # transmission rate
    beta_arr = np.arange(0.1, 1, 0.01) #np.arange(0.04, 0.09, 0.01)
    alpha_arr = np.arange(0.2, 1, 0.01) #np.arange(0.005, 0.011, 0.001)

    n_runs = 10 #50
    times = []
    for beta in tqdm.tqdm(beta_arr):
        for alpha in alpha_arr:
            for seed in range(n_runs):
                start_time = time.time()
                res = network_model.simulate(beta=beta, gamma=gamma, 
                                             delta=delta, 
                                             init_inf_frac=init_inf_frac, 
                                             init_rec_frac=(1-alpha),
                                             tmax=tmax)
                end_time = time.time()
                times.append(end_time-start_time)
                
                sample = [beta, gamma, delta, init_inf_frac, alpha] + res.daily_incidence
                dataset.loc[len(dataset)] = sample
                

                save_seir_df(res, pop, ntype, seed,
                             beta, gamma, delta,
                             init_inf_frac, alpha)
    return times, dataset    


if __name__ == '__main__':


    ntype = 'ba'
    pop = 100000  
    res, dataset = create_all(ntype, pop)
    pd.DataFrame(res).to_csv(f'../sim_data/time_{ntype}_{pop}.csv')
    pd.DataFrame(dataset).to_csv(f'../sim_data/data_{ntype}_{pop}.csv')
    
