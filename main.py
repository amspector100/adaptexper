#!/usr/bin/env python 

import knockadapt
from knockadapt.knockoff_stats import group_lasso_LCD, calc_nongroup_LSM#, group_lasso_LSM

import numpy as np
import pandas as pd
from scipy import stats
import scipy.cluster.hierarchy as hierarchy

import seaborn as sns
import matplotlib.pyplot as plt

import os
import experiments


# GLOBALS ---------------------------------- 
# I think it's better to modify here
# than with compiler flags, since these kwargs
# are extremely complicated.
seed = 110
sample_kwargs = {'coeff_size':10, 
                 'method':'AR1', 
                 'a':5, 
                 'b':1}
S_kwargs = {
    'objective':'norm', 'norm_type':2, 'verbose':True, 'sdp_verbose':False
}
S_methods = [('ASDP_auto', {'method':'ASDP'}), 
             ('ASDP3', {'method':'ASDP', 'alpha':3,}),
             ('ASDP5', {'method':'ASDP', 'alpha':5,})]
link_methods = ['average']*len(S_methods)

n = 100
p = 200
q = 0.25
num_datasets = 3


def plot_measurement_type(melted_results, 
                          meas_type = 'power',
                          fname = None,
                          hline = None):
    """ Plotting and saving function """
    import warnings
    warnings.filterwarnings("ignore")
    from plotnine import *
    warnings.simplefilter("always")

    results = melted_results.loc[melted_results['measurement'] == meas_type]
    results = results.rename(columns = {'value':meas_type})

    g1 = (
        ggplot(results, aes(
            x = 'feature_fn', y = meas_type, fill = 'split_type')
        )
        + geom_boxplot(position='dodge')
        + facet_grid('~link_method')
    )
    if hline is not None:
        g1 = g1 + hline
    
    if fname is not None:
        g1 = g1 + labs(title = fname)
        fname1 = fname + '_' + meas_type + '_v1.SVG'
        g1.save(fname1)
    print(g1)
    
    g2 = (
        ggplot(results, aes(
            x = 'feature_fn', y = meas_type, fill = 'split_type')
        )
        + stat_summary(geom = 'col', position=position_dodge(width=0.9))
        + stat_summary(geom = "errorbar", fun_data = 'mean_cl_boot',
                        position=position_dodge(width=0.9))
        + facet_grid('~link_method')
    )
    
    if hline is not None:
        g2 = g2 + hline
        
    if fname is not None:
        g2 = g2 + labs(title = fname)
        fname2 = fname + '_' + meas_type + '_v2.SVG'
        g2.save(fname2)        
    print(g2)



def main(n, p, 
         q = 0.25, 
         seed = seed,
         num_datasets = 5,
         sample_kwargs = sample_kwargs,
         plot = False):
    """ Main """
    
    # Generate corr_matrix, Q
    np.random.seed(seed)
    X0, y0, beta, Q, corr_matrix = knockadapt.graphs.sample_data(
        n = n, p = p, **sample_kwargs
    )
    
    # Run method comparison function
    output = experiments.compare_methods(
        corr_matrix, 
        beta, 
        Q = Q, 
        n = n,
        q = q, 
        S_methods = S_methods,
        feature_fns = {'LSM':calc_nongroup_LSM, 'group_LCD':group_lasso_LCD},
                       #'group_LSM':group_lasso_LSM},
        link_methods = link_methods,
        S_kwargs = S_kwargs,
        num_data_samples = num_datasets,
    )

    melted_results, oracle_results, S_matrixes = output
    id_vars = ['link_method', 'feature_fn', 'split_type', 'measurement']
    # method_means = melted_results.groupby(id_vars)['value'].mean().reset_index()
    
    # Construct a (long) file name
    fname = f"figures/ASDP/seed{seed}_n{n}_p{p}_N{num_datasets}/"
    if not os.path.exists(fname):
        os.makedirs(fname)
    sample_string = [
        ('').join([k.replace('_', ''), str(sample_kwargs[k])]) for k in sample_kwargs
    ]
    sample_string = ('_').join(sample_string)
    fname += sample_string
    
    # Save CSV
    fname_csv = fname + '.csv'
    melted_results.to_csv(fname_csv)
    
    # Plot and save
    if plot:

        plot_measurement_type(melted_results, 
                              meas_type = 'power', 
                              fname = fname)
        hline = geom_hline(aes(yintercept = q), 
                           linetype="dashed",
                           color = "red")
        plot_measurement_type(melted_results, 
                              meas_type = 'fdr',
                              hline = hline,
                              fname = fname)        


if __name__ == '__main__':
    main(n = n,
         p = p,
         seed = seed, 
         num_datasets = num_datasets,
         q = q,
         sample_kwargs = sample_kwargs)