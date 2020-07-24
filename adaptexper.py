import os
import sys
import time
import datetime
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
from knockadapt import knockoff_stats as kstats
from knockadapt import utilities
from knockadapt.knockoff_filter import KnockoffFilter

import warnings
import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as hierarchy
import pandas as pd

import itertools
from multiprocessing import Pool
from functools import partial

# Global: the set of antisymmetric functions we use
PAIR_AGGS = ['cd', 'sm']#, 'scd']
DEFAULT_q = 0.2


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
	elif isinstance(v, str):
		if v.lower() in ['yes', 'true', 't', 'y', '1']:
			return True
		elif v.lower() in ['no', 'false', 'f', 'n', '0']:
			return False
	return v

def obj2int(v):
	try:
		v = float(v)
		if v.is_integer():
			v = int(v)
		return v
	except: 
		return str2bool(v)

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

def Z2selections(Z, groups, q, **kwargs):
	
	# Calculate W statistics
	W = kstats.combine_Z_stats(Z, groups, **kwargs)

	# Calculate selections 
	T = kstats.data_dependent_threshhold(W=W, fdr=q)
	selected_flags = (W >= T).astype("float32")
	return selected_flags, W

def selection2power(
	selections,
	group_nonnulls,
	group_sizes
):
	# Calculate fdp, power
	fdp = np.sum(selections*(1-group_nonnulls))
	power = np.sum(selections*group_nonnulls/group_sizes)
	epower = np.sum(selections / group_sizes)

	# Normalize
	fdp = fdp / max(1, np.sum(selections))

	# TODO: think if we should normalize this
	power = power / max(1, np.sum(group_nonnulls))
	epower = epower / max(1, np.sum(group_nonnulls))
	return power, epower, fdp

def single_cutoff_statistics(
	p,
	num_groups,
	seed,
	split_type,
	rec_prop,
	cutoff,
	m,
	groups,
	beta,
	group_sizes,
	group_nonnulls,
	knockoff_filter=None,
	**filter_kwargs
):
	"""
	Note that m is the number of groups.
	"""
	# Padding for array format
	padding = np.zeros((p-m))
	padding[:] = np.nan

	# Knockoff filter for this cutoff
	# (Can be passed in)
	filter_kwargs['groups'] = groups
	if knockoff_filter is None:
		knockoff_filter = KnockoffFilter()
		knockoff_filter.forward(
			**filter_kwargs
		)

	# Extract info
	Z = knockoff_filter.Z
	W = knockoff_filter.W
	selections = knockoff_filter.selected_flags
	score = knockoff_filter.score
	score_type = knockoff_filter.score_type
	power, epower, fdp = selection2power(
		selections=selections, 
		group_nonnulls=group_nonnulls,
		group_sizes=group_sizes,
	)

	row = [
		# Seed and identifying info
		seed,
		split_type,
		cutoff,
		rec_prop,
		m,
		# Measurements
		power,
		epower,
		fdp,
		score, 
		score_type,
		# Statistics / dgp
		*groups.tolist(),
		*beta.tolist(),
		*np.concatenate([np.around(W, 4), padding]).tolist(),
		*np.around(Z[0:p], 4).tolist(),
		*np.around(Z[p:], 4).tolist(),
		*np.concatenate([selections, padding]).tolist()
	]
	return row

def single_dataset_cutoff_statistics(
	seed,
	Sigma,
	beta,
	reps,
	num_processes,
	sample_kwargs,
	filter_kwargs,
	time0,
	num_groups,
	cutoffs,
	all_groups,
	S_matrices,
	rec_props,
	split_types,
):

	# Sample data, record time, resample beta if beta is None
	p = Sigma.shape[0]
	localtime = time.time()
	np.random.seed(seed)
	X, y, beta, _, _ = knockadapt.graphs.sample_data(
		corr_matrix=Sigma,
		beta=beta,
		**sample_kwargs
	)
	n = X.shape[0]

	# FDR target level
	if 'q' in filter_kwargs:
		q = filter_kwargs['q']
		filter_kwargs['fdr'] = filter_kwargs.pop('q')
	else:
		q = DEFAULT_q
		filter_kwargs['fdr'] = q

	# Now we loop through the cutoffs 
	cols = ['seed', 'split_type', 'cutoff', 'rec_prop', 'm']
	cols += ['power', 'epower', 'fdp', 'score', 'score_type']
	cols += [f'group{i}' for i in range(1, p+1)]
	cols += [f'beta{i}' for i in range(1, p+1)]
	cols += [f'W{i}' for i in range(1, p+1)]
	cols += [f'Z{i}' for i in range(1, p+1)]
	cols += [f'tildeZ{i}' for i in range(1, p+1)]
	cols += [f'selection{i}' for i in range(1, p+1)] 
	output = pd.DataFrame(
		columns = cols
	)
	counter = 0
	# Regular knockoff filter
	for cutoff, groups, m in zip(cutoffs, all_groups, num_groups):

		# Process groups
		group_sizes = utilities.calc_group_sizes(groups)
		group_nonnulls = knockadapt.utilities.fetch_group_nonnulls(
			beta, groups
		)

		# Add S-matrix to filter_kwargs
		if 'knockoff_kwargs' in filter_kwargs:
			filter_kwargs['knockoff_kwargs']['S'] = S_matrices[m][-1]
		else:
			filter_kwargs['knockoff_kwargs'] = {'S':S_matrices[m][-1]}

		# Regular filter
		if split_types[0] is None or 'regular' in split_types:
			row = single_cutoff_statistics(
				p=p,
				num_groups=num_groups,
				seed=seed,
				split_type='regular',
				rec_prop=-1,
				cutoff=cutoff,
				m=m,
				beta=beta,
				group_sizes=group_sizes,
				group_nonnulls=group_nonnulls,
				# Filter kwargs
				X=X, 
				y=y, 
				mu=np.zeros(p),
				Sigma=Sigma, 
				groups=groups,
				**filter_kwargs
			)
			output.loc[counter] = row
			counter += 1

		# Split / recycled knockoff filters:
		for rec_prop in rec_props:

			# Split knockoff filter, WITHOUT recycling
			# S-matrix should NOT account for recycling
			filter_kwargs['knockoff_kwargs'] = {'S':S_matrices[m][-1]}
			if split_types[0] is None or 'split' in split_types:
				row = single_cutoff_statistics(
					p=p,
					num_groups=num_groups,
					seed=seed,
					split_type='split',
					rec_prop=rec_prop, # This does NOT cause recycling, it's for logging
					cutoff=cutoff,
					m=m,
					beta=beta,
					group_sizes=group_sizes,
					group_nonnulls=group_nonnulls,
					# Filter kwargs
					X=X[:int(rec_prop * n)], 
					y=y[:int(rec_prop * n)], 
					mu=np.zeros(p),
					Sigma=Sigma, 
					groups=groups,
					**filter_kwargs
				)
				output.loc[counter] = row
				counter += 1

			# Second half of split knockoff filter
			if split_types[0] is None or 'split2' in split_types:
				row = single_cutoff_statistics(
					p=p,
					num_groups=num_groups,
					seed=seed,
					split_type='split2',
					rec_prop=rec_prop,  # This does NOT cause recycling, it's for logging
					cutoff=cutoff,
					m=m,
					beta=beta,
					group_sizes=group_sizes,
					group_nonnulls=group_nonnulls,
					# Filter kwargs
					X=X[int(rec_prop * n):], 
					y=y[int(rec_prop * n):], 
					mu=np.zeros(p),
					Sigma=Sigma, 
					groups=groups,
					**filter_kwargs
				)
				output.loc[counter] = row
				counter += 1


			# Split knockoff filter, WITH recycling
			if split_types[0] is None or 'recycled' in split_types:
				output.loc[counter] = single_cutoff_statistics(
					p=p,
					num_groups=num_groups,
					seed=seed,
					split_type='recycled',
					rec_prop=rec_prop,
					cutoff=cutoff,
					m=m,
					beta=beta,
					group_sizes=group_sizes,
					group_nonnulls=group_nonnulls,
					# Filter kwargs
					X=X, 
					y=y, 
					mu=np.zeros(p),
					Sigma=Sigma, 
					groups=groups,
					recycle_up_to=int(rec_prop * n),
					**filter_kwargs
				)
				counter += 1

			# New S-matrix which accounts for recycling
			if 'recycled_new_s' in split_types:
				filter_kwargs['knockoff_kwargs']['S'] = S_matrices[m][rec_prop]
				output.loc[counter] = single_cutoff_statistics(
					p=p,
					num_groups=num_groups,
					seed=seed,
					split_type='recycled_new_s',
					rec_prop=rec_prop,
					cutoff=cutoff,
					m=m,
					beta=beta,
					group_sizes=group_sizes,
					group_nonnulls=group_nonnulls,
					# Filter kwargs
					X=X, 
					y=y, 
					mu=np.zeros(p),
					Sigma=Sigma, 
					groups=groups,
					recycle_up_to=int(rec_prop * n),
					**filter_kwargs
				)
				counter += 1

			# # Precycled / recycled filters
			# knockoff_kwargs = filter_kwargs['knockoff_kwargs'].copy()
			# knockoff_kwargs['S'] = S_matrices[m][-1]
			# kbi = knockadapt.adaptive.KnockoffBicycle(
			# 	knockoff_kwargs=knockoff_kwargs, fixedX=False,
			# )
			# kbi.forward(
			# 	X=X,
			# 	y=y,
			# 	groups=groups,
			# 	mu=np.zeros(p),
			# 	Sigma=Sigma,
			# 	rec_prop=rec_prop,
			# 	**filter_kwargs
			# )
			# for split_type, kf in zip(
			# 	['precycled', 'recycled'],
			# 	[kbi.prefilter, kbi.refilter],
			# ):
			# 	output.loc[counter] = single_cutoff_statistics(
			# 		p=p,
			# 		num_groups=num_groups,
			# 		seed=seed,
			# 		split_type=split_type,
			# 		rec_prop=rec_prop,
			# 		cutoff=cutoff,
			# 		m=m,
			# 		group_sizes=group_sizes,
			# 		group_nonnulls=group_nonnulls,
			# 		# Filter kwargs
			# 		knockoff_filter=kf,
			# 	)
			# 	counter += 1

	if n == MAXIMUM_N and seed % 10 == 0:
		print(f"Finished with seed {seed}, took {time.time() - localtime}")

	return output

def calc_cutoff_statistics(
	time0,
	Sigma,
	beta,
	num_groups,
	cutoffs,
	all_groups,
	reps=1,
	num_processes=1,
	sample_kwargs={},
	filter_kwargs={},
	seed_start=0,
	S_matrices={},
	rec_props=[0.5],
	split_types=[None],
):

	# Fetch nonnulls
	p = Sigma.shape[0]
	# Sample data reps times and calc power/fdp
	partial_sd_cutoff_stats = partial(
		single_dataset_cutoff_statistics, 
		Sigma=Sigma,
		beta=beta,
		reps=reps,
		num_processes=num_processes,
		sample_kwargs=sample_kwargs,
		filter_kwargs=filter_kwargs,
		time0=time0,
		num_groups=num_groups,
		cutoffs=cutoffs,
		all_groups=all_groups,
		S_matrices=S_matrices,
		rec_props=rec_props,
		split_types=split_types,
	)
	# (Possibly) apply multiprocessing
	all_inputs = list(range(seed_start, seed_start+reps))
	num_processes = min(len(all_inputs), num_processes)
	all_outputs = apply_pool(
		func = partial_sd_cutoff_stats,
		all_inputs = all_inputs,
		num_processes=num_processes
	)

	# Extract output and return
	final_out = pd.concat(all_outputs, sort=True)
	return final_out

def analyze_resolution_powers(
	Sigma,
	beta,
	dgp_number,
	filter_kwargs,
	sample_kwargs,
	fstat_kwargs,
	reps,
	num_processes,
	seed_start,
	time0,
	rec_props,
	split_types,
):
	"""
	Computes W-statistics and powers for various resolutions
	using sample-splitting, sample-splitting with recycling,
	precycling, and no splitting at all. 
	At present, precycling is limited to the lasso statistic.
	"""
	global MAXIMUM_N # A hack to allow for better logging

	# Infer p and set n defaults
	p = Sigma.shape[0]
	sample_kwargs['p'] = [p]
	if 'n' not in sample_kwargs:
		sample_kwargs['n'] = [
			int(p/4), int(p/2), int(p/1.5), int(p), int(2*p), int(4*p)
		]
	MAXIMUM_N = max(sample_kwargs['n']) # Helpful for logging

	# Construct/iterate cartesian product of sample, filter, fstat kwargs
	sample_keys, sample_product = dict2keyproduct(sample_kwargs)
	filter_keys, filter_product = dict2keyproduct(filter_kwargs)
	fstat_keys, fstat_product = dict2keyproduct(fstat_kwargs)

	# Create links, correlation tree, cutoffs
	link = knockadapt.graphs.create_correlation_tree(
		Sigma, method='average'
	)
	cutoffs = knockadapt.adaptive.create_cutoffs(
		link=link, reduction=5, max_size=20
	)
	all_groups = [hierarchy.fcluster(link, cutoff, criterion='distance') for cutoff in cutoffs]
	num_groups = [np.unique(groups).shape[0] for groups in all_groups]

	# Precompute S-matrices
	S_matrices = {m:{} for m in num_groups}
	for m, groups in zip(num_groups, all_groups):
		rec_props_to_construct = [-1]
		if 'recycled_new_s' in split_types:
			rec_props_to_construct = rec_props_to_construct + rec_props
		for rec_prop in rec_props_to_construct:
			print(f"Generating S matrix for num_groups={m}, rec_prop={rec_prop} at time={time.time()-time0}")
			S_EQ = knockadapt.knockoffs.equicorrelated_block_matrix(
				Sigma=Sigma, groups=groups
			)
			_, group_S_MCV = knockadapt.knockoffs.gaussian_knockoffs(
				X=np.random.randn(10, p),
				Sigma=Sigma,
				groups=groups,
				init_S=S_EQ,
				method='mcv',
				return_S=True, 
				max_epochs=250,
				rec_prop=max(0, rec_prop),
			)
			S_matrices[m][rec_prop] = group_S_MCV
		print([(x, np.diag(S_matrices[m][x])) for x in S_matrices[m]])
	print(f"Finished with S-matrices at {time.time() - time0}")

	### Calculate power of knockoffs for the cutoffs
	all_outputs = []
	for filter_vals in filter_product:
		filter_vals = list(filter_vals)
		new_filter_kwargs = {
			key:val for key, val in zip(filter_keys, filter_vals)
		}

		# In high dimensional cases or binomial cases,
		# don't fit OLS.
		if 'feature_stat' in new_filter_kwargs:
			fstat = new_filter_kwargs['feature_stat']
		else:
			fstat = 'lasso'

		for sample_vals in sample_product:
			sample_vals = list(sample_vals)
			new_sample_kwargs = {
				key:val for key, val in zip(sample_keys, sample_vals)
			}

			# Don't run OLS in certain cases
			if fstat == 'ols':
				if new_sample_kwargs['n'] < 2*new_sample_kwargs['p']:
					continue
				if 'y_dist' in new_sample_kwargs:
					if new_sample_kwargs['y_dist'] == 'binomial':
						continue
			if fstat == 'dlasso':
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

				# Power/FDP for the S methods
				output = calc_cutoff_statistics(
					Sigma=Sigma,
					beta=beta,
					reps=reps,
					num_processes=num_processes,
					sample_kwargs=new_sample_kwargs,
					filter_kwargs=new_filter_kwargs,
					seed_start=seed_start,
					time0=time0,
					num_groups=num_groups,
					cutoffs=cutoffs,
					all_groups=all_groups,
					S_matrices=S_matrices,
					rec_props=rec_props,
					split_types=split_types,
				)
				for key in sample_keys:
					if key in output.columns:
						print(f"Weird, sample key {key} is already in output")
					output[key] = new_sample_kwargs[key]
				for key in filter_keys:
					if key in output.columns:
						print(f"Weird, filter key {key} is already in output")
					output[key] = new_filter_kwargs[key]
				for key in fstat_keys:
					if key in output.columns:
						print(f"Weird, fstat key {key} is already in output")
					output[key] = new_fstat_kwargs[key]
				all_outputs.append(output)

	result_df = pd.concat(all_outputs, sort=True, ignore_index=True)
	result_df['dgp_number'] = dgp_number
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

	The --description argument must be the last argument, as 
	all arguments after the --description argument will be ignored.
	"""
	
	# Get rid of script name
	args = args[1:]

	# Initialize kwargs constructor
	special_keys = ['reps', 'num_processes', 'seed_start', 'description', 'resample_beta', 'rec_props', 'split_types']
	key_types = ['dgp', 'sample', 'filter', 'fstat']
	all_kwargs = {ktype:{} for ktype in key_types}
	key = None
	key_type = None # One of dgp, sample, filter
	description_index = None # At the end, can write description

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
			if key in special_keys:
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

			# Friendly reminder
			if key == 'feature_stat_fn':
				raise ValueError("feature_stat_fn is depreciated, use feature_stat")

			# Description
			if key == 'description':
				description_index = i
				break

		# Parse values
		if i % 2 == 1:

			# Check if it's a list written out
			if arg[0] == '[':
				# Strip brackets, whitspace, and split by commas
				value = arg.replace('[', '').replace(']', '')
				value = value.replace(' ', '').split(',')
				# Try to process
				value = [obj2int(v) for v in value]

			# Check if it's start{num}end{num}numvals{num}
			elif arg[0:5] == 'start':
				start = float(arg.replace('start', '').split('end')[0])
				end = float(arg.split('end')[-1].split('numvals')[0])
				numvals = int(arg.split('numvals')[-1])
				value = np.linspace(
					start, end, numvals
				)

			# Apply obj2int (preserves strings, infers floats, bools, ints)
			else:
				value = obj2int(arg)


			all_kwargs[key_type][key] = val2list(value)

	# Parse description 
	description = ''
	if description_index is not None:
		description += (' ').join(args[description_index+1:])
	description += f'\n \n Other arguments were: {args[0:description_index]}'
	all_kwargs['sample']['description'] = description

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

	# Make sure pair_aggs is not being duplicated
	if 'pair_agg' in fstat_kwargs:
		raise ValueError("Many pair_aggs will be analyzed anyway. Do not add this as a fstat_kwarg.")
	# Some common errors I make
	if 'y_dist' in dgp_kwargs:
		raise ValueError("y_dist ought to be in sample_kwargs")

	# Parse some special non-graph kwargs
	reps = fetch_kwarg(sample_kwargs, 'reps', default=[1])[0]
	num_processes = fetch_kwarg(sample_kwargs, 'num_processes', default=[5])[0]
	seed_start = fetch_kwarg(sample_kwargs, 'seed_start', default=[0])[0]
	description = fetch_kwarg(sample_kwargs, 'description', default='')
	resample_beta = fetch_kwarg(sample_kwargs, 'resample_beta', default=[False])[0]
	rec_props = fetch_kwarg(sample_kwargs, 'rec_props', default=[0.5])
	split_types = fetch_kwarg(
		sample_kwargs,
		'split_types',
		default=[None]
	)

	print(f"DGP kwargs are {dgp_kwargs}")
	print(f"Sample kwargs are {sample_kwargs}")
	print(f"Filter kwargs are {filter_kwargs}")
	print(f"Fstat kwargs are {fstat_kwargs}")
	print(f"Description is {description.split('Other arguments were:')[0]}")

	# Create output paths
	today = str(datetime.date.today())
	hour = str(datetime.datetime.today().time())
	hour = hour.replace(':','-').split('.')[0]
	output_path = f'data/adaptv6/{today}/{hour}'
	all_key_types = ['dgp', 'sample', 'filter', 'fstat']
	all_kwargs = [dgp_kwargs,sample_kwargs, filter_kwargs, fstat_kwargs]

	for key_type,kwargs in zip(all_key_types, all_kwargs):
		output_path += f'/{key_type}_'
		keys = sorted([k for k in kwargs])
		for k in keys:
			path_val = ('').join(str(kwargs[k]).split(' '))
			output_path += f'{k}{path_val}_'

	# Put it all together and ensure directory exists
	output_path += f'seedstart{seed_start}_reps{reps}_results.csv'
	beta_path = output_path.split('.csv')[0] + '_betas.csv'
	S_path = output_path.split('.csv')[0] + '_S.csv'
	description_path = f'data/adaptv6/{today}/{hour}/' + 'description.txt'
	output_dir = os.path.dirname(output_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	print(f"Output path is {output_path}")

	# Save description
	with open(description_path, 'w') as thefile:
		thefile.write(description)	

	# Initialize final final output
	all_results = pd.DataFrame()

	# Loop through DGP parameters
	dgp_keys = sorted([key for key in dgp_kwargs])
	dgp_components = []
	for key in dgp_keys:
		dgp_kwargs[key] = val2list(dgp_kwargs[key])
		dgp_components.append(dgp_kwargs[key])
	dgp_product = itertools.product(*dgp_components)

	# p sadly does need to have 1 value for a variety of reasons 
	if 'p' in dgp_kwargs:
		if len(dgp_kwargs['p']) > 1:
			raise ValueError(f"Must have only one value of p, not {dgp_kwarys[p]}")
		else:
			p = dgp_kwargs['p'][0]
	else:
		p = 100

	dgp_number = 0
	beta_df = pd.DataFrame(columns=np.arange(1, p+1, 1))
	beta_df.index.name = 'dgp_number'

	for dgp_vals in dgp_product:

		# Create DGP using dgp kwargs
		dgp_vals = list(dgp_vals)
		new_dgp_kwargs = {key:val for key,val in zip(dgp_keys, dgp_vals)}
		print(f"DGP kwargs are now: {new_dgp_kwargs}")
		np.random.seed(110)
		_, _, beta, _, Sigma = knockadapt.graphs.sample_data(
			**new_dgp_kwargs
		)

		# Cache beta
		if not resample_beta:
			beta_df.loc[dgp_number] = beta
			beta_df.to_csv(beta_path)

		# Create results
		result = analyze_resolution_powers(
			Sigma=Sigma,
			beta=beta if not resample_beta else None,
			dgp_number=dgp_number,
			sample_kwargs=sample_kwargs,
			filter_kwargs=filter_kwargs,
			fstat_kwargs=fstat_kwargs,
			reps=reps,
			num_processes=num_processes,
			seed_start=seed_start,
			time0=time0,
			rec_props=rec_props,
			split_types=split_types
		)
		for key in dgp_keys:
			if key in sample_kwargs:
				continue
			result[key] = new_dgp_kwargs[key]
		all_results = all_results.append(
			result, 
			ignore_index = True
		)
		all_results.to_csv(output_path)

		# Increment dgp number
		dgp_number += 1

	relevant_cols = [c for c in all_results.columns if c[0] not in ['W', 'Z']]
	relevant_cols = [c for c in relevant_cols if 'tilde' not in c]
	relevant_cols = [c for c in relevant_cols if 'selection' not in c]
	print(all_results.groupby(['cutoff', 'split_type'])[['power', 'epower', 'fdp']].mean())
	return all_results

if __name__ == '__main__':

	time0 = time.time()
	MAXIMUM_N = 0
	main(sys.argv)


