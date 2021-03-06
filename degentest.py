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
from knockadapt.knockoff_filter import mx_knockoff_filter

import warnings
import numpy as np
import pandas as pd

from multiprocessing import Pool
from functools import partial

def fetch_kwarg(kwargs, key, default=None):
	""" Utility function for parsing """
	if key in kwargs:
		return kwargs.pop(key)
	else:
		return default

def str2bool(v):
	""" Helper function, converts strings to boolean vals""" 
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise ValueError('Boolean value expected.')

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

def single_dataset_power_fdr(
	seed,
	Sigma,
	beta,
	groups,
	S,
	q=0.2,
	rec_up_to=None,
	normalize=True,
	feature_stat_fn=None,
	feature_stat_kwargs={},
	**kwargs
):
	# Fetch groups
	p = Sigma.shape[0]
	if groups is None:
		groups = np.arange(1, p+1, 1)
	group_nonnulls = knockadapt.utilities.fetch_group_nonnulls(
		beta, groups
	)
	
	np.random.seed(seed)
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

	# Infer feature_stat_fn and its kwargs
	if feature_stat_fn is None:
		feature_stat_fn=knockadapt.knockoff_stats.lasso_statistic
		feature_stat_kwargs['y_dist'] = y_dist
		if 'group_lasso' not in feature_stat_kwargs:
			feature_stat_kwargs['group_lasso'] = False

	# Run MX knockoff filter
	selections = mx_knockoff_filter(
		X=X, 
		y=y, 
		Sigma=Sigma, 
		groups=groups,
		recycle_up_to=rec_up_to,
		feature_stat_fn=feature_stat_fn,
		feature_stat_kwargs=feature_stat_kwargs,
		knockoff_kwargs={'S':S, 'verbose':False},
		fdr=q,
	)

	# Calculate fdp, power, return
	fdp = np.sum(selections*(1-group_nonnulls))
	power = np.sum(selections*group_nonnulls)
	# Possibly divide by # of non-nulls
	if normalize:
		fdp = fdp/max(1, np.sum(selections))
		power = power/max(1, np.sum(group_nonnulls))

	return (power, fdp)

def calc_power_and_fdr(
	Sigma,
	beta,
	S,
	groups=None,
	q=0.2,
	reps=100,
	rec_up_to=None, 
	num_processes=1,
	**kwargs
):
	
	# Fetch nonnulls
	p = Sigma.shape[0]
	# Sample data reps times and calc power/fdp
	partial_sd_power_fdr = partial(
		single_dataset_power_fdr, 
		rec_up_to=rec_up_to, 
		Sigma=Sigma,
		beta=beta,
		groups=groups,
		S=S,
		q=q, 
		**kwargs
	)
	# (Possibly) apply multiprocessing
	all_inputs = list(range(reps))
	num_processes = min(len(all_inputs), num_processes)
	all_outputs = apply_pool(
		func = partial_sd_power_fdr,
		all_inputs = all_inputs,
		num_processes=num_processes
	)
	# Extract output and return
	powers = [x[0] for x in all_outputs]
	fdps = [x[1] for x in all_outputs]
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
		*args,
		rec_up_to=rec_up_to,
		n=n,
		**kwargs
	)

	# Pass args for scaling
	kwargs['S'] = (1-prop)*kwargs['S'].copy()
	out_scale = calc_power_and_fdr(
		*args,
		n=n,
		rec_up_to=None,
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
	q=0.2,
	test_recycling=False,
	**kwargs
	):
	
	p = Sigma.shape[0]
	if n_values is None:
		n_values = [
			int(p/4), int(p/2), int(p/1.5), int(p), int(2*p), int(4*p)
		]
	if prop_rec is None:
		prop_rec = np.arange(0, 8, 1)/10

	# Check if we're going to run 
	# S_opt with recycling
	if test_recycling:
		psgd_prop_rec = prop_rec
	else:
		psgd_prop_rec = [0]

	# Final output
	counter = 0
	result_df = pd.DataFrame(
		columns = ['power', 'fdp', 'n', 'method', 'prop_rec']
	)

	### Calculate power of optimal S using ncvx solver
	### for different types of recycling
	groups = np.arange(1, p+1, 1)
	for prop in psgd_prop_rec:
		# Optimal S
		opt = knockadapt.nonconvex_sdp.NonconvexSDPSolver(
			Sigma=Sigma,
			groups=groups,
			init_S=S,
			rec_prop=prop
		)
		opt_S = opt.optimize(max_epochs=750)
		print(opt_S)
		print(f'Finished computing opt_S matrix for prop_rec = {prop}, time is {time.time() - time0}')
		for n in n_values:
			rec_up_to = int(prop*n)
			powers, fdps = calc_power_and_fdr(
				Sigma=Sigma,
				beta=beta,
				S=opt_S,
				groups=groups,
				q=q,
				reps=reps,
				rec_up_to=rec_up_to, 
				num_processes=num_processes,
				n=n,
				**kwargs
			)
			for power, fdp in zip(powers, fdps):
				result_df.loc[counter] = [power, fdp, n, 'psgd', prop]
				counter += 1

	### Calculate the curve over recycling/scaling
	# Helper function which will be used for multiprocessing 
	partial_calc_power_and_fdr = partial(
		wrap_calc_power_and_fdr, 
		Sigma=Sigma, beta=beta, S=S, q=q, reps=reps, **kwargs
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

	# Add to pandas output
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


	# Create kwargs
	kwargs = parse_args(args)
	print(f"Args were {args}")
	print(f"Sample kwargs are {kwargs}")

	# Parse some special non-graph kwargs
	reps = fetch_kwarg(kwargs, 'reps', default=50)
	num_processes = fetch_kwarg(kwargs, 'num_processes', default=5)
	test_recycling = fetch_kwarg(kwargs, 'test_recycling', default=False)
	test_recycling = str2bool(test_recycling)

	# Curve parameter - create values
	curve_param = fetch_kwarg(kwargs, 'curve_param', default='None')
	num_param_values = fetch_kwarg(kwargs, 'num_param_values', default=5)
	param_min = fetch_kwarg(kwargs, 'param_min', default=0)
	param_max = fetch_kwarg(kwargs, 'param_max', default=1)
	if curve_param == "None":
		param_values = [None]
	else:
		param_values = np.linspace(
			param_min, param_max, num_param_values
		)
		param_values = np.around(param_values, 3)

	# Create output path
	output_dir = 'data/degentest/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	output_path = output_dir + f'curve_param{curve_param}_'
	kwarg_keys = sorted([k for k in kwargs])
	for k in kwarg_keys:
		output_path += f'{k}{kwargs[k]}_'
	if test_recycling:
		output_path += 'rectest_'
	output_path += 'results.csv'


	# Loop through curve parameters
	all_results = pd.DataFrame()
	for param_val in param_values:

		# Add to kwargs (unless it's a dummy)
		if curve_param != "None":
			print(f'Analyzing {curve_param} value {param_val} at time {time.time() - time0}')
			kwargs[curve_param] = param_val

		# Create DGP
		np.random.seed(110)
		_, _, beta, _, Sigma = knockadapt.graphs.sample_data(
			**kwargs
		)

		# Create groups and solve SDP
		p = Sigma.shape[0]
		groups = np.arange(1, p+1, 1)
		print(f'Computing S matrix, time is {time.time() - time0}')
		S = knockadapt.knockoffs.solve_group_SDP(Sigma, groups=groups)
		print(f'Finished computing S matrix, time is {time.time() - time0}')

		# Create results
		result = analyze_degen_solns(
			Sigma=Sigma,
			beta=beta,
			S=S,
			n_values=None,
			prop_rec=None,
			reps=reps,
			num_processes=num_processes,
			test_recycling=test_recycling,
			**kwargs
		)
		if curve_param != "None":
			result[curve_param] = param_val
		all_results = all_results.append(
			result, 
			ignore_index = True
		)
		all_results.to_csv(output_path)

	return all_results

if __name__ == '__main__':

	time0 = time.time()
	main(sys.argv)