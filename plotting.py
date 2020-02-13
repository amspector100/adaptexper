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
		fname1 = fname + '_' + meas_type + '_v1.JPG'
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
		fname2 = fname + '_' + meas_type + '_v2.JPG'
		g2.save(fname2)        
	#print(g2)
	return None


def plot_csv(csv_path, q = None):
	""" Convenience function to prevent Odyssey from having to plot"""
	fname = csv_path.split('.')[0]
	melted_results = pd.read_csv(csv_path)
	if q is None:
		warnings.warn('No q specified, assuming 0.2 by default')
		q = 0.2
		
	plot_measurement_type(melted_results, 
						  meas_type = 'power', 
						  fname = fname)
	plot_measurement_type(melted_results, 
						  meas_type = 'fdr',
						  yintercept = q,
						  fname = fname) 

def construct_reg_from_oracle(oracle_results, x_axis):
	""" Reconstructs 'regular' type results from oracle results"""

	#,link_method,feature_fn,sample,cutoff,num_groups,variable,value,measurement,split_type,x_axis

	# Step 1: for each oracle type, pick the best cutoff (actual power)
	power_grouping = [x_axis, 'link_method', 'feature_fn', 'oracle_type']
	cutoff_powers = oracle_results.groupby(power_grouping+['cutoff'])['power'].mean().reset_index()
	best_cutoffs = cutoff_powers.groupby(power_grouping)['power'].idxmax().reset_index()
	best_cutoffs['power'] = cutoff_powers.loc[best_cutoffs['power'].values, 'cutoff'].values
	best_cutoffs=best_cutoffs.rename(columns={'power':'best_cutoff'})

	# Step 2: Construct true oracle data 
	oracle_results = pd.merge(
		oracle_results, best_cutoffs, on = power_grouping
	)
	oracle_data = oracle_results.loc[oracle_results['cutoff'] == oracle_results['best_cutoff']]
	oracle_data['split_type'] = oracle_data['oracle_type']

	# Step 3: Construct non-split data
	chosen_cutoff_ids = oracle_results.groupby(
		['sample', x_axis, 'feature_fn', 'oracle_type']
	)['epower'].idxmax().values
	nonsplit_data = oracle_results.loc[chosen_cutoff_ids]
	nonsplit_data['split_type'] = 'nonsplit'

	# Step 4: Construct baseline
	baseline_data = oracle_results.loc[oracle_results['cutoff']==0]
	baseline_data['split_type'] = 'baseline'

	# Step 5: Combine
	reg_data = pd.concat(
		[oracle_data, nonsplit_data, baseline_data], 
		axis=0, 
		ignore_index=True
	)
	# Melt and drop useless columns
	value_vars = ['power', 'epower', 'fdp']
	id_vars = [c for c in reg_data.columns if c not in value_vars]
	reg_data=pd.melt(
		reg_data,
		id_vars=id_vars,
		value_vars=value_vars,
		value_name='value',
		var_name='variable'
	)
	reg_data = reg_data.drop(
		['best_cutoff', 'oracle_type'],
		axis='columns'
	)
	# Rename fdp to fdr
	reg_data['variable'] = reg_data['variable'].str.replace('fdp', 'fdr')

	return reg_data

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
	if os.path.exists(csv_path):
		results = pd.read_csv(csv_path)
		results = results.drop('Unnamed: 0', axis = 'columns')
	else:
		results = None
		print("Note regular data does not exist")

	# Oracle data	
	oracle_path = path + '_oracle.csv'
	if os.path.exists(oracle_path):
		oracle_results = pd.read_csv(oracle_path)
		oracle_results = oracle_results.drop('Unnamed: 0', axis = 'columns')
	else:
		oracle_results = None
		print("Note oracle data does not exist")

	#col_subset = [c for c in results.columns if c != 'sample']
	#results = results.drop_duplicates(col_subset)

	# Get some plotting values
	n_vals = oracle_results['n'].unique()
	if curve_param is not None:
		if n_vals.shape[0] != 1:
			raise ValueError(f"Cannot decide whether to plot power curve along n or {curve_param} axis")
		x_breaks = oracle_results[curve_param].unique()
		x_axis = curve_param
		geom = 'line'
		size = 1
		scale_cts = True
	elif n_vals.shape[0] != 1:
		x_breaks = n_vals
		x_axis = 'n'
		geom = 'line'
		size = 1
		scale_cts = True
	else:
		x_breaks = None
		x_axis = 'split_type'
		geom = 'point'
		size = 5
		scale_cts = False

	# Reconstruct results using oracle
	if results is None:
		results = construct_reg_from_oracle(oracle_results, x_axis=x_axis)

	# Figure out which column contains 'power'/'fdr' only:
	# this is helpful to deal with some legacy graphs
	# (in the newer version, it should be variable)
	if results is not None:
		if 'power' in results['variable'].unique():
			var_column = 'variable'
		else:
			var_column = 'measurement'

		# Plot power, empirical power, and FDR
		warnings.filterwarnings("ignore")
		print('Plotting FDRs, powers')
		for meas_type in results[var_column].unique():

			# Path
			new_path = path + '/' + meas_type + '.JPG'
			dirname = os.path.dirname(new_path)
			if not os.path.exists(dirname):
				os.makedirs(dirname)


			# Subset
			subset = results.loc[results[var_column] == meas_type]
			subset = subset.rename(columns = {'value':meas_type})

			# Plot
			g2 = (
				ggplot(subset, aes(
					x = x_axis, y = meas_type, color = 'split_type', fill = 'split_type')
				)
				+ stat_summary(geom = geom, size = size)
				+ stat_summary(aes(shape = 'split_type'), geom = 'point', size = 2.5)
				+ stat_summary(geom = "errorbar", fun_data = 'mean_cl_boot', width = 0.01)
				+ facet_grid('feature_fn~link_method', scales = 'fixed')
				+ labs(title = new_path)
			)

			if meas_type == 'fdr':
				hline = geom_hline(
					aes(yintercept = q), linetype="dashed", color = "red"
				)
				g2 += hline
			if scale_cts:
				g2 += scale_x_continuous(breaks = x_breaks)


			plotnine.options.figure_size = (10, 8)
			g2.save(new_path)

		# Plot average group sizes and average cutoffs
		print('Plotting groups, cutoffs')
		# At this point, we can't have a discrete x axis
		if x_axis == 'split_type':
			x_axis = 'n'
		for col in ['num_groups', 'cutoff']:

			new_path = path + '/' + col + '.JPG'
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
			)

			if scale_cts:
				g2 += scale_x_continuous(breaks = x_breaks)


			plotnine.options.figure_size = (10, 8)
			g2.save(new_path)

	# Plot oracle powers/fdps by cutoffs
	print("Plotting oracle data")
	feature_fns = oracle_results['feature_fn'].unique()
	for feature_fn in feature_fns:
		subset = oracle_results.loc[oracle_results['feature_fn'] == feature_fn]
		for col in ['cutoff']:
			for meas_type in ['power', 'epower', 'fdp', 'num_groups']:

				new_path = path + f'/all_{col}_' + meas_type + '_' + feature_fn + '.JPG'
				dirname = os.path.dirname(new_path)
				if not os.path.exists(dirname):
					os.makedirs(dirname)

				# Make plotting pretty
				subset[x_axis] = np.around(subset[x_axis], 3)
				subset[col] = np.around(subset[col], 3)

				g2 = (
					ggplot(oracle_results, aes(
						x = col, y = meas_type, color = x_axis, fill = x_axis,
					))
					+ stat_summary(geom = 'point', size = 2.5)
					+ stat_summary(geom = "errorbar", fun_data = 'mean_cl_normal', width = 0.01)
					+ facet_grid(f'oracle_type~{x_axis}')
					+ labs(title = new_path)
				)

				g2.save(new_path)

	if oracle_results['oracle_type'].unique().shape[0] > 1:
		print('------ \n ------ WARNING: THERE ARE MULTIPLE ORACLE TYPES AGGREGATED ------- \n ----------')
		raise ValueError("Too many oracle types present")

	# Round and string-ify cutoffs
	oracle_results['cutoff'] = np.round(oracle_results['cutoff'], 3).astype('str')
	g1 = (
		ggplot(oracle_results, aes(
			x='cutoff', y='epower', fill='cutoff', group='cutoff')
		)
		+ geom_boxplot(position='dodge')
		+ facet_grid(f'feature_fn~{x_axis}')
	)
	new_path = path + f'/epower_dists.JPG'
	g1.save(new_path)

	# Distribution of best cutoff
	chosen_cutoff_ids = oracle_results.groupby(
		['sample', x_axis, 'feature_fn', 'oracle_type']
	)['epower'].idxmax().values
	chosen_cutoffs = oracle_results.loc[chosen_cutoff_ids]

	# Density for each setting
	num_samples = oracle_results['sample'].max() + 1
	cutoff_counts = chosen_cutoffs.groupby(
		[x_axis, 'feature_fn', 'oracle_type'], 
	)['cutoff'].value_counts(normalize=True).unstack().fillna(0).reset_index()
	cutoff_counts = pd.melt(
		cutoff_counts,
		id_vars=[x_axis, 'feature_fn', 'oracle_type'],
		var_name='cutoff',
		value_name='proportion'
	)

	g1 = (
		ggplot(
			cutoff_counts, aes(
			x='cutoff', y='proportion', fill='cutoff',
			)
		)
		+ geom_col(position='dodge')
		+ facet_grid(f'feature_fn~{x_axis}')
	)
	new_path = path + f'/selection_dists.JPG'
	g1.save(new_path)


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

		# Plot if existant
		if not os.path.exists(new_path):
			pass
		else:
			plot_csv(new_path)

	warnings.simplefilter("always")



if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Please specify the path to create plots for')
	else:
		sys.exit(plot_n_curve(sys.argv[1]))