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

import itertools
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

def val2list(val):
	""" Turns discrete values into lists, otherwise returns """
	if not (isinstance(val, list) or isinstance(val, np.ndarray)):
		return [val] 
	return val

def dict2keyproduct(dictionary):
	""" Takes a dictionary mapping to lists
	and returns the sorted list of keys and 
	the cartesian product of each list."""
	keys = sorted([key for key in dictionary])
	components = []
	for key in keys:
		dictionary[key] = val2list(dictionary[key])
		components.append(dictionary[key])
	product = list(itertools.product(*components))
	return keys, list(product)

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

def fetch_competitor_S(
	Sigma,
	groups,
):

	### Special case: detect if Sigma is equicorrelated,
	# in which case we can calculate the solution analytically
	# for ungrouped knockoffs.
	p = Sigma.shape[0]
	if np.unique(groups).shape[0] == p:
		rho = Sigma[0, 1]
		equicorr = rho*np.ones((p, p)) + (1-rho)*np.eye(p)
		if np.all(Sigma==equicorr):
			print(f"Sigma is equicorr (rho={rho}), using analytical solution")
			S_SDP = min(1, 2-2*rho)*np.eye(p)
			#scale_FKTP = (2-2*rho)*(p-np.sqrt(p**2 - p))
			scale_FKTP = (1-rho)
			S_FKTP = scale_FKTP*np.eye(p)
			print(S_FKTP)
			if rho < 0.5:
				return {'sdp':S_SDP, 'ftkp':S_FKTP}
			else:
				S_SDP_perturbed = S_SDP*(0.99)
				return {
				'sdp':S_SDP, 
				'sdp_perturbed':S_SDP_perturbed, 
				'ftkp':S_FKTP
				}



	### Calculate (A)SDP S-matrix 
	if p <= 1000:
		print(f'Computing SDP S matrix, time is {time.time() - time0}')
		S_SDP = knockadapt.knockoffs.solve_group_SDP(Sigma, groups=groups)
	else:
		print(f'Computing ASDP S matrix, time is {time.time() - time0}')
		S_SDP = knockadapt.knockoffs.solve_group_ASDP(
			Sigma=Sigma, 
			groups=groups, 
			max_block=500, 
			num_processes=num_processes,
		)
	print(f'Finished computing S matrix, time is {time.time() - time0}')

	### Calculate FTKP matrix (nonconvex solver)
	opt = knockadapt.nonconvex_sdp.NonconvexSDPSolver(
		Sigma=Sigma,
		groups=groups,
		init_S=S_SDP,
	)
	S_FKTP = opt.optimize(max_epochs=100)
	print(f'Finished computing opt_S matrix, time is {time.time() - time0}')

	return {'sdp':S_SDP, 'ftkp':S_FKTP}

def single_dataset_power_fdr(
	seed,
	Sigma,
	beta,
	groups,
	S,
	q=0.2,
	normalize=True,
	sample_kwargs={},
	filter_kwargs={
		'feature_stat_kwargs':{},
	},
):

	# Fetch groups
	p = Sigma.shape[0]
	if groups is None:
		groups = np.arange(1, p+1, 1)
	group_nonnulls = knockadapt.utilities.fetch_group_nonnulls(
		beta, groups
	)

	# Sample data
	np.random.seed(seed)
	X, y, _, _, _ = knockadapt.graphs.sample_data(
		corr_matrix=Sigma,
		beta=beta,
		**sample_kwargs
	)

	# Run MX knockoff filter
	selections = mx_knockoff_filter(
		X=X, 
		y=y, 
		Sigma=Sigma, 
		groups=groups,
		knockoff_kwargs={'S':S, 'verbose':False},
		fdr=q,
		**filter_kwargs
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
	num_processes=1,
	sample_kwargs={},
	filter_kwargs={},
):
	
	# Fetch nonnulls
	p = Sigma.shape[0]
	# Sample data reps times and calc power/fdp
	partial_sd_power_fdr = partial(
		single_dataset_power_fdr, 
		Sigma=Sigma,
		beta=beta,
		groups=groups,
		S=S,
		q=q, 
		sample_kwargs=sample_kwargs,
		filter_kwargs=filter_kwargs,
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

def analyze_degen_solns(
	Sigma,
	beta,
	groups=None,
	sample_kwargs={},
	filter_kwargs={},
	fstat_kwargs={},
	q=0.2,
	reps=50,
	num_processes=5,
	):
	"""
	:param sample_kwargs: 
	A dictionary. Each key is a sample parameter name, with the value
	being either a single value or a list or array of values. 
	:param filter_kwargs: 
	A dictionary. Each key is a mx_filter parameter name, with the value
	being either a single value or a list or array of values. 
	"""
	
	# Infer p and set n defaults
	p = Sigma.shape[0]
	sample_kwargs['p'] = [p]
	if 'n' not in sample_kwargs:
		sample_kwargs['n'] = [
			int(p/4), int(p/2), int(p/1.5), int(p), int(2*p), int(4*p)
		]
	if groups is None:
		groups = np.arange(1, p+1, 1)

	# Construct/iterate cartesian product of sample, filter, fstat kwargs
	sample_keys, sample_product = dict2keyproduct(sample_kwargs)
	filter_keys, filter_product = dict2keyproduct(filter_kwargs)
	fstat_keys, fstat_product = dict2keyproduct(fstat_kwargs)

	# Initialize final output
	counter = 0
	columns = ['power', 'fdp', 'S_method'] 
	columns += sample_keys + filter_keys + fstat_keys
	result_df = pd.DataFrame(columns=columns)

	# Create competitor S-matrices
	S_matrices = fetch_competitor_S(Sigma, groups)

	### Calculate power of knockoffs for the two different methods
	for filter_vals in filter_product:
		filter_vals = list(filter_vals)
		new_filter_kwargs = {
			key:val for key, val in zip(filter_keys, filter_vals)
		}

		# In high dimensional cases or binomial cases,
		# don't fit OLS.
		if 'feature_stat_fn' in new_filter_kwargs:
			if new_filter_kwargs['feature_stat_fn'] == 'ols':
				ols_flag = True
			else:
				ols_flag = False
		else:
			ols_flag = False

		for sample_vals in sample_product:
			sample_vals = list(sample_vals)
			new_sample_kwargs = {
				key:val for key, val in zip(sample_keys, sample_vals)
			}

			# Don't run OLS in certain cases
			if ols_flag:
				if new_sample_kwargs['n'] < 2*new_sample_kwargs['p']:
					continue
				if 'y_dist' in new_sample_kwargs:
					if new_sample_kwargs['y_dist'] == 'binomial':
						continue

			for fstat_vals in fstat_product:
				# Extract feature-statistic kwargs
				# and place them properly (as a dictionary value
				# of the filter_kwargs)
				fstat_vals = list(fstat_vals)
				new_fstat_kwargs = {
					key:val for key, val in zip(fstat_keys, fstat_vals)
				}
				if 'feature_stat_kwargs' in new_filter_kwargs:
					new_filter_kwargs['feature_stat_kwargs'] = dict(
						new_filter_kwargs['feature_stat_kwargs'],
						**new_fstat_kwargs
					)
				else:
					new_filter_kwargs['feature_stat_kwargs'] = new_fstat_kwargs

				# Loop through competitor methods
				for S_method in S_matrices:

					# Power/FDP for the method
					S = S_matrices[S_method]
					powers, fdps = calc_power_and_fdr(
						Sigma=Sigma,
						beta=beta,
						S=S,
						groups=groups,
						q=q,
						reps=reps,
						num_processes=num_processes,
						sample_kwargs=new_sample_kwargs,
						filter_kwargs=new_filter_kwargs,
					)
					for power, fdp in zip(powers, fdps):
						row = [power, fdp, S_method] 
						row += sample_vals + filter_vals + fstat_vals
						result_df.loc[counter] = row 
						counter += 1

	return result_df

def parse_args(args):
	""" Homemade argument parser 
	Usage: --argname_{dgp/sample/filter} value
	Value should be one of:
		- integer/float
		- arbitrary string
		- Alternatively, string of the form
		"start{num}end{num}numvals{num}" 
		which indicates that the parameter ought 
		to be varied along a linear interpolation
		from the specified start/end with the specified
		number of values.
		- Alternatively, a string like
		"[str1, str2, str3]" which will be parsed as
		["str1", "str2", "str3"]
	
	Note that "reps" and "num_processes" are special kwargs.
	They get placed as sample_kwargs by default but will be
	extracted.
	"""
	
	# Get rid of script name
	args = args[1:]

	# Initialize kwargs constructor
	key_types = ['dgp', 'sample', 'filter', 'fstat']
	all_kwargs = {ktype:{} for ktype in key_types}
	key = None
	key_type = None # One of dgp, sample, filter

	#Parse
	for i, arg in enumerate(args):
		arg = str(arg).lower()

		# Even placements should be keyword
		if i % 2 == 0:
			if str(arg)[0:2] != '--':
				raise ValueError(
					f'{i}th arg ({arg}) should be a kwarg name preceded by "--"'
				)
			key = arg[2:]

			# Check what type of keyword
			if key in ['reps', 'num_processes']:
				key_type = 'sample'
			else:
				key_split = key.split('_')
				key_type = key_split[-1]
				key = ('_').join(key_split[0:-1])

			# Raise error if unexpected key type
			if key_type not in key_types:
				raise ValueError(
					f'{i}th arg ({arg}) has key_type {key_type}, must be one of {key_types}'
				)

		# Parse values
		if i % 2 == 1:
			# Try to parse this as float/int
			try:
				value = float(arg)
				if value.is_integer():
					value = int(value)
			# Try to parse this as a list of values
			except:
				# Check if it's a list written out
				if arg[0] == '[':
					# Strip brackets, whitspace, and split by commas
					value = arg.replace('[', '').replace(']', '')
					value = value.replace(' ', '').split(',')

				# Check if it's start{num}end{num}numvals{num}
				elif arg[0:5] == 'start':
					start = float(arg.replace('start', '').split('end')[0])
					end = float(arg.split('end')[-1].split('numvals')[0])
					numvals = int(arg.split('numvals')[-1])
					value = np.linspace(
						start, end, numvals
					)

				# If all else fails, it's a string!
				else:
					value = arg


			all_kwargs[key_type][key] = val2list(value)

	out = (all_kwargs[key_type] for key_type in key_types)
	return out

def main(args):
	""" Layers of keyword arguments are as follows.
	- dgp_kwargs: kwargs needed to create Sigma (e.g. p, method)
	- sample_kwargs: kwargs needed to sample data (e.g. n)
	- filter_kwargs: kwargs for the knockoff filter. (e.g. recycling)
	- fstat_kwargs: kwargs for the feature statistic. (e.g. pair_agg)
	
	For each of these, keys mapping to iterables like lists or 
	numpy arrays will be varied. See parse_args comments.

	The MAIN constraint is that the same sample_kwargs must be used
	for all dgp_kwargs. E.g., it will be difficult to vary both
	a for an AR1 covariance matrix and delta for an ErdosRenyi covariance
	matrix. The same goes for filter_kwargs abd fstat_kwargs.
	"""


	# Create kwargs
	dgp_kwargs, sample_kwargs, filter_kwargs, fstat_kwargs = parse_args(args)
	print(f"Args were {args}")

	# Parse some special non-graph kwargs
	reps = fetch_kwarg(sample_kwargs, 'reps', default=[50])[0]
	num_processes = fetch_kwarg(sample_kwargs, 'num_processes', default=[5])[0]
	print(f"DGP kwargs are {dgp_kwargs}")
	print(f"Sample kwargs are {sample_kwargs}")
	print(f"Filter kwargs are {filter_kwargs}")
	print(f"fstat kwargs are {fstat_kwargs}")


	# Create output path
	output_dir = 'data/degentestv2/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	output_path = output_dir
	for kwargs in [dgp_kwargs,sample_kwargs, filter_kwargs]:
		keys = sorted([k for k in kwargs])
		for k in keys:
			print(kwargs[k])
			path_val = kwargs[k][0] if len(kwargs[k]) == 1 else "varied"
			print(kwargs, k, path_val)
			output_path += f'{k}{path_val}_'
	output_path += 'results.csv'
	print(output_path)

	# Initialize final final output
	all_results = pd.DataFrame()

	# Loop through DGP parameters
	dgp_keys = sorted([key for key in dgp_kwargs])
	dgp_components = []
	for key in dgp_keys:
		dgp_kwargs[key] = val2list(dgp_kwargs[key])
		dgp_components.append(dgp_kwargs[key])
	dgp_product = itertools.product(*dgp_components)

	for dgp_vals in dgp_product:

		# Create DGP using dgp kwargs
		dgp_vals = list(dgp_vals)
		new_dgp_kwargs = {key:val for key,val in zip(dgp_keys, dgp_vals)}
		print(f"DGP kwargs are now: {new_dgp_kwargs}")
		np.random.seed(110)
		_, _, beta, _, Sigma = knockadapt.graphs.sample_data(
			**new_dgp_kwargs
		)

		# Create results
		result = analyze_degen_solns(
			Sigma,
			beta,
			groups=None,
			sample_kwargs=sample_kwargs,
			filter_kwargs=filter_kwargs,
			fstat_kwargs=fstat_kwargs,
			q=0.2,
			reps=reps,
			num_processes=num_processes,
		)
		for key in dgp_keys:
			result[key] = new_dgp_kwargs[key]
		print(result)
		all_results = all_results.append(
			result, 
			ignore_index = True
		)
		all_results.to_csv(output_path)

	return all_results

if __name__ == '__main__':

	time0 = time.time()
	main(sys.argv)