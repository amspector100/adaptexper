#!/usr/bin/env python 

import sys
import os
import argparse
import time

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
	sys.stdout.write(f'Knockoff dir is {knockoff_directory}\n')
	sys.path.insert(0, os.path.abspath(knockoff_directory))
	import knockadapt
	from knockadapt.knockoff_stats import group_lasso_LCD
		
import experiments


def str2bool(v):
	""" Helper function, converts strings to boolean vals""" 
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def get_sample_string(sample_kwargs):
	""" Creates a string (for filenames) representing the sample kwargs""" 
	sample_string = [
		('').join([k.replace('_', ''), str(sample_kwargs[k])]) for k in sample_kwargs
	]
	sample_string = ('_').join(sample_string)
	return sample_string


def main(args):
	""" Simulates power and FDR of various knockoff methods """

	description = 'Simulates power, fdr of various knockoff methods'
	parser = argparse.ArgumentParser(description = description)

	parser.add_argument('--n', dest = 'n',
						type=int, 
						help='Number of observations (default: 100). If n = 0, will run 6 sims between p/4 and 4p',
						default = 100)

	parser.add_argument('--p', dest = 'p',
						type=int, 
						help='Number of covariates/features (default: 50)',
						default = 50)

	parser.add_argument('--q', dest = 'q',
						type=float, 
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

	parser.add_argument('--nosdp', dest = 'nosdp',
						type=str, 
						help='If true, do NOT use full semidefinite programming (default: False)',
						default = False)

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
						help='a parameter when sampling AR1 correlations from Beta(a,b) (default: 1)',
						default = 1)

	parser.add_argument('--b', dest = 'b',
						type=float,
						help='b parameter when sampling AR1 correlations from Beta(a,b) (default: 1)',
						default = 1)

	parser.add_argument('--gamma', dest = 'gamma',
						type=int,
						help='gamma parameter when sampling daibarber2016 graphs (default: 0)',
						default = 0)

	parser.add_argument('--coef', dest = 'coef',
						type=float,
						help='Size of true coefficients relative to noise level (default: 10)',
						default = 10)

	parser.add_argument('--recompute', dest = 'recompute',
					type=bool,
					help='If true, recompute non S matrix related results (default: False)',
					default = False)

	parser.add_argument('--scache', dest = 'scache',
					type=bool,
					help='If true, only compute S matrices, do nothing else (default: False)',
					default = False)

	parser.add_argument('--numprocesses', dest = 'numprocesses',
					type=int,
					help='How many processes to use in multiprocessing package (default: 8)',
					default = 8)

	parser.add_argument('--splitoracles', dest = 'splitoracles',
					type=bool,
					help='To test the split oracles (default: False)',
					default = False)

	parser.add_argument('--reduction', dest = 'reduction',
					type=int,
					help='How many different groupings/cutoffs to test (default: 10)',
					default = 10)

	parser.add_argument('--pyglmnet', dest = 'pyglmnet',
					type=str,
					help='Whether to use pyglmnet as the lasso backend (default: True)',
					default = 'True')

	parser.add_argument('--curve_param', dest = 'curve_param',
					type=str,
					help="""Vary this parameter to create a power/fdr curve with
							10 evenly spaced values on a logarithmic/linear scale.
							MUST be a valid argument to pass to sample_kwargs.
							(default: '', no curve)',
							""",
					default = '')

	parser.add_argument('--param_min', dest = 'param_min',
					type=float,
					help='Minimum value of parameter in power/fdr curve (default: 0)',
					default = 0)

	parser.add_argument('--param_max', dest = 'param_max',
					type=float,
					help='Maximum value of parameter in power/fdr curve (default: 1)',
					default = 1)

	parser.add_argument('--param_spacing', dest = 'param_spacing',
					type=str,
					help='How to space parameter in power/fdr curve (one of linear/log)',
					default = 'linear')

	args = parser.parse_args()
	args.pyglmnet = str2bool(args.pyglmnet)
	sys.stdout.write(f'Parsed args are {args} \n')

	# Retreive values
	n = args.n
	p = args.p
	q = args.q
	seed = args.seed
	num_datasets = args.num_datasets
	plot = args.plot
	scache = args.scache
	recompute = args.recompute
	num_processes = args.numprocesses
	splitoracles = args.splitoracles
	reduction = args.reduction
	use_pyglm = args.pyglmnet
	curve_param = args.curve_param.lower()

	# Generate S methods
	S_kwargs = {'objective':'norm', 
				'norm_type':2, 
				'verbose':True, 
				'sdp_verbose':False}

	if not args.nosdp:
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
	if args.covmethod.lower() == 'daibarber2016':
		if curve_param != 'gamma':
			sample_kwargs['gamma'] = args.gamma
	else:
		if curve_param != 'coeff_size':
			sample_kwargs['coeff_size'] = args.coef

	# Initialize save directories
	all_fname = f"data/v4/"
	sample_string = get_sample_string(sample_kwargs)
	all_fname += sample_string
		
	# Possibly signify that we're doing a power curve
	# for some parameter...
	if curve_param != '':
		all_fname += f'_curve{curve_param}'

	# Add seed, p, q, N information
	all_fname += f'/seed{seed}_p{p}/'
	if not os.path.exists(all_fname):
		os.makedirs(all_fname)
	all_fname += f'q{q}_N{num_datasets}'
	all_fname_csv = all_fname + '.csv'
	all_fname_oracle_csv = all_fname + '_oracle.csv'

	# Fixed n 
	if n != 0:
		ns = [n]
	else:
		if curve_param == '': 
			ns = [p/4, p/2, p, 2*p, 4*p]
			ns = [int(n) for n in ns]
		else:
			raise ValueError(f"Cannot set n = 0 and have curve_param {curve_param}")

	# Possibly prepare power curve
	if curve_param != '':
		if args.param_spacing.lower() == 'linear':
			curve_vals = np.around(np.linspace(
				args.param_min, args.param_max, 10
			), 3)
		elif args.param_spacing.lower() == 'log':
			curve_vals = np.around(np.logspace(
				args.param_min, args.param_max, 10
			), 3)
		else:
			raise ValueError(f"param_spacing must be 'linear' or 'log', not {args.param_spacing}")
		ns = ns * len(curve_vals)
	else:
		# Dummy variables which we won't use
		curve_vals = [None for _ in ns]

	# Timing
	time0 = time.time()

	# Possibly generate singular corr_matrix, Q
	if curve_param == '':
		np.random.seed(seed)
		_, _, beta, Q, corr_matrix = knockadapt.graphs.sample_data(
			n = ns[0], p = p, **sample_kwargs
		)
	else:
		beta = None
		corr_matrix = None
		Q = None
	
	# Initialize all result dataframe
	all_results = pd.DataFrame()
	all_oracle_results = pd.DataFrame()

	# Loop through ns: no need to parallelize this yet
	# since each n takes quite a while
	for n, curve_val in zip(ns, curve_vals):

		if curve_val is not None:
			sample_kwargs[curve_param] = curve_val

		# Create filename, check that we haven't already done this computation
		sample_string = get_sample_string(sample_kwargs)
		fname = f"data/v2/{sample_string}/seed{seed}_p{p}_n{n}"
		if not os.path.exists(fname):
			os.makedirs(fname)
		fname = fname + f'/q{q}_N{num_datasets}'
		fname_csv = fname + '.csv'
		fname_oracle_csv = fname + '_oracle.csv'

		# Possibly use cached data
		if not recompute and os.path.exists(fname_csv) and os.path.exists(fname_oracle_csv):
			sys.stdout.write(f'Using cached data for n = {n}, {curve_param} = {curve_val}\n')
			oracle_results = pd.read_csv(fname_oracle_csv, index_col = 0)
			melted_results = pd.read_csv(fname_csv, index_col = 0)

		# Else, it's time to compute!
		else:
			sys.stdout.write(f'Running simulation for n = {n}, {curve_param} = {curve_val}\n')

			# Run method comparison function - note the 
			# S matrices should be cached 
			link_methods = ['average']*max(1, len(S_methods))
			output = experiments.compare_methods(
				corr_matrix, 
				beta, 
				Q = Q, 
				n = n,
				p = p,
				q = q, 
				S_methods = S_methods,
				feature_fns = {'group_LCD':group_lasso_LCD},
				feature_fn_kwargs = {'group_LCD':{'use_pyglm':use_pyglm}},
				link_methods = link_methods,
				S_kwargs = S_kwargs,
				num_data_samples = num_datasets,
				sample_kwargs = sample_kwargs,
				time0 = time0,
				seed = seed,
				scache_only = scache,
				num_processes = num_processes,
				compute_split_oracles = splitoracles,
				reduction = reduction
			)
			# Possibly exit if we only need to compute S matrices
			if scache:
				return None

			# Unpack and cache
			melted_results, oracle_results, _ = output
			melted_results.to_csv(fname_csv)
			oracle_results.to_csv(fname_oracle_csv)
			
			# Possibly plot, usually not though - delete this later
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

		# Aggregate with all other data
		melted_results['n'] = n
		oracle_results['n'] = n
		if curve_val is not None:
			melted_results[curve_param] = curve_val
			oracle_results[curve_param] = curve_val

		all_results = pd.concat(
			[all_results, melted_results], axis = 'index'
		)
		all_oracle_results = pd.concat(
			[all_oracle_results, oracle_results], axis = 'index'
		)

		# Save
		all_results.to_csv(all_fname_csv)
		all_oracle_results.to_csv(all_fname_oracle_csv)

	return all_results, all_oracle_results

if __name__ == '__main__':
	if '--profile' in sys.argv:
		sys.argv.remove('--profile')

		# Profile
		import cProfile
		cProfile.run('main(sys.argv)', 'profile')

		# Analyze
		import pstats
		p = pstats.Stats('profile')
		p.strip_dirs().sort_stats('cumulative').print_stats(50)
		
	else:
		sys.exit(main(sys.argv))
