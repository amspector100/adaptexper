import os
import sys
import time
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
from knockadapt import knockoff_stats
from knockadapt.utilities import random_permutation_inds, chol2inv
from knockadapt.knockoff_filter import mx_knockoff_filter

import warnings
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import scipy.cluster.hierarchy as hierarchy

import seaborn as sns
import matplotlib.pyplot as plt


import experiments
from smatrices import construct_S_path

from multiprocessing import Pool
from functools import partial


def apply_pool(func, all_inputs, num_processes):
	""" Utility function"""

	# Don't use the pool object if n-processes is 1
	if num_processes == 1:
		all_outputs = []
		for inp in all_inputs:
			all_outputs.append(func(inp))
	else:
		with Pool(num_processes) as thepool:
			all_outputs = thepool.map(func, all_inputs)

	return all_outputs


def calc_power_and_fdr(
	rec_up_to, 
	Sigma,
	beta,
	S,
	groups=None,
	q=0.2,
	reps=100,
	num_processes=1,
	**kwargs
):
	
	np.random.seed(110)

	# Fetch nonnulls
	p = Sigma.shape[0]
	if groups is None:
		groups = np.arange(1, p+1, 1)
	group_nonnulls = knockadapt.utilities.fetch_group_nonnulls(beta, groups)

	# Container for fdps
	fdps = []
	powers = []

	# Sample data reps times
	for j in range(reps):
		X, y, _, _, _ = knockadapt.graphs.sample_data(
			corr_matrix=Sigma,
			beta=beta,
			**kwargs
		)

		# Infer y_dist
		if 'y_dist' in kwargs:
			y_dist = kwargs['y_dist']
		else:
			y_dist = 'gaussian'

		# Run MX knockoff filter
		selections = mx_knockoff_filter(
			X=X, 
			y=y, 
			Sigma=Sigma, 
			groups=groups,
			recycle_up_to=rec_up_to,
			feature_stat_kwargs={'group_lasso':False, 'y_dist':y_dist},
			knockoff_kwargs={'S':S, 'verbose':False},
			fdr=q,
		)

		# Calculate fdp
		fdp = np.sum(selections*(1-group_nonnulls))/max(1, np.sum(selections))
		fdps.append(fdp)
		
		# Calculate power
		power = np.sum(selections*group_nonnulls)/max(1, np.sum(group_nonnulls))
		powers.append(power)
		
	return np.array(powers), np.array(fdps)

def wrap_calc_power_and_fdr(
	n_prop_tuple,
	*args,
	**kwargs
):
	""" Wrap above function to deal with multiprocessing """

	# Extract n and proportion
	n, prop = n_prop_tuple
	rec_up_to = int(n*prop)

	# Pass args for recycling
	out_rec = calc_power_and_fdr(
		rec_up_to,
		*args,
		n = n,
		**kwargs
	)

	# Pass args for scaling
	kwargs['S'] = (1-prop)*kwargs['S'].copy()
	out_scale = calc_power_and_fdr(
		None,
		*args,
		n = n,
		**kwargs
	)

	return (out_rec, out_scale, n, prop)


def analyze_degen_solns(
	Sigma,
	beta,
	S,
	n_values=None,
	prop_rec=None,
	reps=50,
	num_processes=5,
	**kwargs
	):
	
	p = Sigma.shape[0]
	if n_values is None:
		n_values = [
			int(p/4), int(p/2), int(p/1.5), int(p), int(2*p), int(4*p)
		]
	if prop_rec is None:
		prop_rec = np.arange(0, 11, 1)/10

	# Helper function which will be used for multiprocessing -------
	partial_calc_power_and_fdr = partial(
		wrap_calc_power_and_fdr, 
		Sigma=Sigma, beta=beta, S=S, q=0.2, reps=reps, **kwargs
	)

	# Construct arguments
	all_input_tuples = []
	for n in n_values:
		for prop in prop_rec:
			all_input_tuples.append((n, prop))


	# Apply pool
	num_processes = min(len(all_input_tuples), num_processes)
	all_outputs = apply_pool(
		func = partial_calc_power_and_fdr,
		all_inputs = all_input_tuples,
		num_processes=num_processes
	)

	# Construct pandas output
	result_df = pd.DataFrame(
		columns = ['power', 'fdp', 'n', 'method', 'prop_rec']
	)
	counter = 0
	for (out_rec, out_scale, n, prop) in all_outputs:

		# Add recycling powers/fdps
		rec_powers, rec_fdps = out_rec
		for power, fdp in zip(rec_powers, rec_fdps):
			result_df.loc[counter] = [power, fdp, n, 'rec', prop]
			counter += 1

		# Add scaling powers/fdps
		scale_powers, scale_fdps = out_scale
		for power, fdp in zip(scale_powers, scale_fdps):
			result_df.loc[counter] = [power, fdp, n, 'scale', prop]
			counter += 1

	return result_df

def parse_args(args):
	""" Homemade argument parser """
	
	# Get rid of script name
	args = args[1:]

	# Initialize kwargs constructor
	kwargs = {}
	key = None

	#Parse
	for i, arg in enumerate(args):
		arg = str(arg).lower()

		# Even placements should be keyword
		if i % 2 == 0:
			if str(arg)[0:2] != '--':
				raise ValueError(
					f'{i}th argument ({arg}) should be a sample kwarg name preceded by "--"'
				)
			key = arg[2:]
		if i % 2 == 1:
			try:
				value = float(arg)
				if value.is_integer():
					value = int(value)
				kwargs[key] = value
			except:
				kwargs[key] = arg

	return kwargs

def main(args):

	# Add kwargs
	kwargs = parse_args(args)
	print(f"Sample kwargs are {kwargs}")

	# Parse reps and kwargs
	if 'reps' in kwargs:
		reps = kwargs.pop('reps')
	else:
		reps = 50
	if 'num_processes' in kwargs:
		num_processes = kwargs.pop('num_processes')
	else:
		num_processes = 5

	# Create DGP
	np.random.seed(110)
	_, _, beta, _, Sigma = knockadapt.graphs.sample_data(
		**kwargs
	)

	# Create output path
	output_dir = 'data/degentest/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	kwarg_keys = sorted([k for k in kwargs])
	for k in kwarg_keys:
		output_dir += f'{k}{kwargs[k]}'
	output_dir += 'results.csv'

	# Create groups and solve SDP
	p = Sigma.shape[0]
	groups = np.arange(1, p+1, 1)
	S = knockadapt.knockoffs.solve_group_SDP(Sigma, groups=groups)

	# Create results
	result = analyze_degen_solns(
		Sigma=Sigma,
		beta=beta,
		S=S,
		n_values=None,
		prop_rec=None,
		reps=reps,
		num_processes=num_processes,
		**kwargs
	)

	result.to_csv(output_dir)
	return result

if __name__ == '__main__':

	main(sys.argv)