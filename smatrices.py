""" File which handles caching of S matrices """

import os
import numpy as np

# Global list of kwargs which aren't important for S generation
beta_kwargs = ['coeff_size', 'sparsity', 'k']

def construct_S_path(p, seed, cutoff,
					 link_method, sample_kwargs,
					 roundto = 5):

	# Round cutoff and remove useless string
	cutoff = str(round(cutoff, roundto))
	for arg in beta_kwargs:
		if arg in sample_kwargs:
			sample_kwargs.pop(arg)

	# Turn kwargs into a string
	sample_string = [
		('').join([k.replace('_', ''), str(sample_kwargs[k])]) for k in sample_kwargs
	]
	sample_string = ('_').join(sample_string)

	# Construct path
	path = f'data/S/p{p}_seed{seed}_{sample_string}/cutoff{cutoff}_link{link_method}'
	path = path + '.txt'

	# Return
	return path

def cache_S_matrix(S, p, seed, cutoff,
				   link_method, sample_kwargs,
				   roundto = 5):

	# Get path
	path = construct_S_path(
		p, seed, cutoff, link_method, sample_kwargs, roundto = roundto
	)

	# Create directory
	dirname = os.path.dirname(path)
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	# Save S as csvprint(path)
	np.savetxt(path, S)

def load_S_matrix(p, seed, cutoff,
				  link_method, sample_kwargs,
				  roundto = 5):
	""" If S has been cached, load it. 
	Otherwise, return None."""

	# Get path
	path = construct_S_path(
		p, seed, cutoff, link_method, sample_kwargs, roundto = roundto
	)

	# Check if file exists
	if os.path.exists(path):
		return np.loadtxt(path, dtype = float)
	else:
		return None