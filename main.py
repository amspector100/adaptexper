#!/usr/bin/env python 

import sys
import os
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import scipy.cluster.hierarchy as hierarchy
import matplotlib.pyplot as plt

try:
	import knockadapt
except ImportError:
	# This is super hacky, will no longer be necessary after development ends
	file_directory = os.path.dirname(os.path.abspath(__file__))
	parent_directory = os.path.split(file_directory)[0]
	knockoff_directory = parent_directory + '/knockadapt'
	sys.path.insert(0, os.path.abspath(knockoff_directory))
	import knockadapt
	from knockadapt.knockoff_stats import group_lasso_LCD, calc_nongroup_LSM#, group_lasso_LSM
	
import experiments



def main(args):
	""" Simulates power and FDR of various knockoff methods """

	description = 'Simulates power, fdr of various knockoff methods'
	parser = argparse.ArgumentParser(description = description)

	parser.add_argument('--n', dest = 'n',
						type=int, 
						help='Number of observations (default: 100)',
						default = 100)

	parser.add_argument('--p', dest = 'p',
						type=int, 
						help='Number of covariates/features (default: 50)',
						default = 50)

	parser.add_argument('--q', dest = 'q',
						type=int, 
						help='Level of FDR control (default: .25)',
						default = 0.25)

	parser.add_argument('--seed', dest = 'seed',
						type=int, 
						help='Random seed for reproducibility (default: 110)',
						default = 110)

	parser.add_argument('--num_datasets', dest = 'num_datasets',
						type=int, 
						help='Number of datasets to average over (default: 5)',
						default = 5)

	parser.add_argument('--plot', dest = 'plot',
						type=bool, 
						help='If true, plot results via plotnine (default: False)',
						default = False)

	parser.add_argument('--sdp', dest = 'sdp',
						type=str, 
						help='If true, use full semidefinite programming (default: True)',
						default = True)

	parser.add_argument('--asdp', dest = 'asdp',
						type=str, 
						help='If true, use asdp instead of/in addition to SDP (default: False)',
						default = False)

	parser.add_argument('--covmethod', dest = 'covmethod',
						type=str, 
						help='Method for generating cov matrix: one of "AR1" (default) or "ErdosRenyi"',
						default = 'AR1')

	parser.add_argument('--a', dest = 'a',
						type=float,
						help='a parameter when samplign AR1 correlations from Beta(a,b) (default: 1)',
						default = 1)

	parser.add_argument('--b', dest = 'b',
						type=float,
						help='b parameter when samplign AR1 correlations from Beta(a,b) (default: 1)',
						default = 1)

	parser.add_argument('--coef', dest = 'coef',
						type=float,
						help='Size of true coefficients relative to noise level (default: 10)',
						default = 10)


	args = parser.parse_args()

	# Retreive values
	n = args.n
	p = args.p
	q = args.q
	seed = args.seed
	num_datasets = args.num_datasets
	plot = args.plot

	# Generate S methods
	S_kwargs = {'objective':'norm', 
				'norm_type':2, 
				'verbose':True, 
				'sdp_verbose':False}

	if args.sdp:
		S_methods = [('SDP', {'method':'SDP'})]
	else:
		S_methods = []
	if args.asdp:
		S_methods.append(('ASDP_auto', {'method':'ASDP'}))
	else:
		pass

	# Sample Kwargs
	sample_kwargs = {}
	sample_kwargs['method'] = args.covmethod
	if args.covmethod.lower() == 'ar1':
		sample_kwargs['a'] = args.a
		sample_kwargs['b'] = args.b
	sample_kwargs['coeff_size'] = args.coef
	
	# Generate corr_matrix, Q
	np.random.seed(seed)
	X0, y0, beta, Q, corr_matrix = knockadapt.graphs.sample_data(
		n = n, p = p, **sample_kwargs
	)
	
	link_methods = ['average']*max(1, len(S_methods))


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
	
	# Construct a (long) file name
	fname = f"figures/v2/seed{seed}_n{n}_p{p}_N{num_datasets}/"
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

		# Avoid messy plotnine dependency if not plotting
		from plotting import plot_measurement_type
		plot_measurement_type(melted_results, 
								meas_type = 'power', 
								fname = fname)

		plot_measurement_type(melted_results, 
								meas_type = 'fdr',
								yintercept = q,
								fname = fname)        

	return melted_results 

if __name__ == '__main__':
	sys.exit(main(sys.argv))