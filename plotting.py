import os
import sys
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import plotnine
from plotnine import *
warnings.simplefilter("always")

def plot_measurement_type(melted_results, 
						  meas_type = 'power',
						  fname = None,
						  yintercept = None):
	""" Plotting and saving function """


	if 'power' in melted_results['variable'].unique():
		var_column = 'variable'
	else:
		var_column = 'measurement'

	results = melted_results.loc[melted_results[var_column] == meas_type]
	results = results.rename(columns = {'value':meas_type})

	g1 = (
		ggplot(results, aes(
			x = 'feature_fn', y = meas_type, fill = 'split_type')
		)
		+ geom_boxplot(position='dodge')
		+ facet_grid('~link_method')
	)
	if yintercept is not None:
		hline = geom_hline(
			aes(yintercept = yintercept), linetype="dashed", color = "red"
		)
		g1 = g1 + hline
	
	if fname is not None:
		g1 = g1 + labs(title = fname)
		fname1 = fname + '_' + meas_type + '_v1.SVG'
		g1.save(fname1)
	#print(g1)
	
	g2 = (
		ggplot(results, aes(
			x = 'feature_fn', y = meas_type, fill = 'split_type')
		)
		+ stat_summary(geom = 'col', position=position_dodge(width=0.9))
		+ stat_summary(geom = "errorbar", fun_data = 'mean_cl_boot',
						position=position_dodge(width=0.9))
		+ facet_grid('~link_method')
	)
	
	if yintercept is not None:
		hline = geom_hline(
			aes(yintercept = yintercept), linetype="dashed", color = "red"
		)
		g2 = g2 + hline
		
	if fname is not None:
		g2 = g2 + labs(title = fname)
		fname2 = fname + '_' + meas_type + '_v2.SVG'
		g2.save(fname2)        
	#print(g2)
	return None


def plot_csv(csv_path, q = None):
	""" Convenience function to prevent Odyssey from having to plot"""
	fname = csv_path.split('.')[0]
	melted_results = pd.read_csv(csv_path)
	if q is None:
		warnings.warn('No q specified, assuming 0.25 by default')
		q = 0.25
		
	plot_measurement_type(melted_results, 
						  meas_type = 'power', 
						  fname = fname)
	plot_measurement_type(melted_results, 
						  meas_type = 'fdr',
						  yintercept = q,
						  fname = fname)  

def plot_n_curve(path): 
	""" Creates FDR/power curve plots as n varies """


	csv_path = path + '.csv'
	q = float(path.split('q')[-1].split('_')[0])
	print(f"Parsed q is {q}")
	if 'curve' in path:
		curve_param = path.split('curve')[-1].split('/')[0]
		print(f"Parsed curve_param is {curve_param}")
	else:
		curve_param = None

	# Read data
	results = pd.read_csv(csv_path)
	results = results.drop('Unnamed: 0', axis = 'columns')
	col_subset = [c for c in results.columns if c != 'sample']
	#results = results.drop_duplicates(col_subset)

	# Get some plotting values
	n_vals = results['n'].unique()
	if curve_param is not None:
		if n_vals.shape[0] != 1:
			raise ValueError(f"Cannot decide whether to plot power curve along n or {curve_param} axis")
		x_breaks = results[curve_param].unique()
		x_axis = curve_param
	else:
		x_breaks = n_vals
		x_axis = 'n'

	# Figure out which column contains 'power'/'fdr' only:
	# this is helpful to deal with some legacy graphs
	# (in the newer version, it should be variable)
	if 'power' in results['variable'].unique():
		var_column = 'variable'
	else:
		var_column = 'measurement'

	# Plot power, empirical power, and FDR
	warnings.filterwarnings("ignore")
	print('Plotting FDRs, powers')
	for meas_type in results[var_column].unique():

		# Path
		new_path = path + '/' + meas_type + '.SVG'
		dirname = os.path.dirname(new_path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)


		# Subset
		subset = results.loc[results[var_column] == meas_type]
		subset = subset.rename(columns = {'value':meas_type})

		# Plot
		g2 = (
			ggplot(subset, aes(
				x = x_axis, y = meas_type, color = 'split_type')
			)
			+ stat_summary(geom = 'line')
			+ stat_summary(aes(shape = 'split_type'), geom = 'point', size = 2.5)
			+ stat_summary(geom = "errorbar", fun_data = 'mean_cl_boot', width = 0.01)
			+ facet_grid('feature_fn~link_method')
			+ labs(title = new_path)
			+ scale_x_continuous(breaks = x_breaks)
		)

		if meas_type == 'fdr':
			hline = geom_hline(
				aes(yintercept = q), linetype="dashed", color = "red"
			)
			g2 += hline

		plotnine.options.figure_size = (10, 8)
		g2.save(new_path)

	# Plot average group sizes and average cutoffs
	print('Plotting groups, cutoffs')
	for col in ['num_groups', 'cutoff']:

		new_path = path + '/' + col + '.SVG'
		dirname = os.path.dirname(new_path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		g2 = (
			ggplot(subset, aes(
				x = x_axis, y = col, color = 'split_type')
			)
			+ stat_summary(geom = 'line')
			+ stat_summary(aes(shape = 'split_type'), geom = 'point', size = 2.5)
			+ stat_summary(geom = "errorbar", fun_data = 'mean_cl_normal', width = 0.01)
			+ facet_grid('feature_fn~link_method')
			+ labs(title = new_path)
			+ scale_x_continuous(breaks = x_breaks)
		)
		plotnine.options.figure_size = (10, 8)
		g2.save(new_path)

	# Plot for individual n for each n
	for n in n_vals:

		# Path
		n = int(n)
		new_path = path.replace('v4', 'v2')
		split_new_path = new_path.split('_p')
		if len(split_new_path) != 2:
			raise IndexError('Could not analyze path name properly')
		new_path = ''.join([split_new_path[0], f'_n{n}_p', split_new_path[1]])
		new_path = new_path + '.csv'

		# Plot
		plot_csv(new_path)

	warnings.simplefilter("always")



if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Please specify the path to create plots for')
	else:
		sys.exit(plot_n_curve(sys.argv[1]))