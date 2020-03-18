import functools
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

def unpackage_W_stats(oracle_results):

	# Infer p
	numcols = [c for c in oracle_results.columns.tolist() if str(c).isdigit()]
	p = int(numcols[-1]) + 1
	print(f"My guess is that p is {p}")

	# Create Ws, group_sizes, nulls
	Ws = pd.DataFrame()
	nnulls = pd.DataFrame()
	gsizes = pd.DataFrame()
	cols = [str(x) for x in range(p)]
	for col in cols:

		# Split and retrieve
		splits = oracle_results[col].str.split(pat="|", n=0, expand=True)
		if splits.shape[1] != 3:
			raise ValueError("Groupsize information is missing")

		# Add to big dfs
		Ws[col] = splits[2].astype('float32')
		nnulls[col] = splits[1].str.contains('non-')
		gsizes[col] = splits[0].astype('float32')

	# Return
	return Ws.values, nnulls.values, gsizes.values

def construct_reg_from_oracle(oracle_results, x_axis, q, partition_stats = None, modify_oracle = True):
	""" Reconstructs 'regular' type results from oracle results
	if modify_oracle: then adds wbinstat column to oracle. """

	#,link_method,feature_fn,sample,cutoff,num_groups,variable,value,measurement,split_type,x_axis

	# Step 0: Possibly Add weighted binomial stat data to oracle_results
	# Uses signs but (almost) no noise
	reduce_var = int(1e9)
	num_partitions = oracle_results['cutoff'].unique().shape[0]
	if partition_stats is None:
		Ws, nnulls, gsizes = unpackage_W_stats(oracle_results)
		oracle_results[f'wbinstat_nonoise'] = knockadapt.adaptive.weighted_binomial_feature_stat(
			Ws, gsizes, eps=0.05, delta=0.01, reduce_var=reduce_var,
			num_partitions=num_partitions
		)
		oracle_results[f'apprx_epower'] = knockadapt.adaptive.apprx_epower(
			Ws, gsizes, eps=0.05, delta=0.01, reduce_var=reduce_var,
			num_partitions=num_partitions
		)

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
	
	# Print some stuff
	mean_epowers = oracle_results.groupby(power_grouping+['cutoff'])['epower'].mean()
	print(mean_epowers)

	# Step 3: Construct non-split data
	chosen_cutoff_ids = oracle_results.loc[oracle_results['oracle_type'] == 'oracle'].groupby(
		['sample', x_axis, 'feature_fn', 'link_method']
	)['epower'].idxmax().values
	nonsplit_data = oracle_results.loc[chosen_cutoff_ids]
	nonsplit_data['split_type'] = 'nonsplit'

	# Step 4: Construct sample-split
	if 'split_oracle' in oracle_results['oracle_type'].unique():

		# Select maximum epower for each sample using SPLIT oracle
		spl_grouper = ['sample', x_axis, 'feature_fn', 'link_method']
		spl_oracle_results = oracle_results.loc[oracle_results['oracle_type'] == 'split_oracle']
		chosen_cutoff_ids = spl_oracle_results.groupby(spl_grouper)['epower'].idxmax().values
		chosen_cutoffs = spl_oracle_results.loc[chosen_cutoff_ids, spl_grouper+['cutoff']]
		chosen_cutoffs['spl_selection_flag'] = 1

		# Then, use recycling oracle
		rec_data = oracle_results.loc[oracle_results['oracle_type'] == 'rec_oracle']
		rec_data = pd.merge(
			left=rec_data, 
			right=chosen_cutoffs, 
			on=spl_grouper+['cutoff'], 
			how='inner'
		)
		rec_data['split_type'] = 'sample-split'

	else:
		rec_data = pd.DataFrame()
		

	# Step 5: Construct baseline
	baseline_data = oracle_results.loc[
		(oracle_results['cutoff']==0) & (oracle_results['oracle_type']=='oracle')
	]
	baseline_data['split_type'] = 'baseline'

	# Step 6: Weighted-binomial stuff
	ns_results = oracle_results.loc[oracle_results['oracle_type'] == 'oracle']

	# Partition statistics
	pstat_data = pd.DataFrame()
	def add_to_pstat_data(pstat_data, feature):
		""" Add a partition statistic feature to data """

		# Select cutoffs with highest partition stat
		chosen_cutoff_ids = ns_results.groupby(
			['sample', x_axis, 'feature_fn', 'link_method']
		)[feature].idxmax().values

		# Add to pstat_data
		to_add = ns_results.loc[chosen_cutoff_ids]
		to_add['split_type'] = feature
		pstat_data = pd.concat(
			[pstat_data, to_add], 
			ignore_index=True,
			sort=True
		)

		# Return
		return pstat_data


	# Unpackage and calculate epowers, check 
	if partition_stats is None:

		Ws, nnulls, gsizes = unpackage_W_stats(ns_results)
		epowers = knockadapt.knockoff_stats.calc_epowers(
			Ws=Ws, non_nulls=nnulls, group_sizes=gsizes, fdr=q
		)
		print(epowers)
		print(ns_results['epower'])

		# Apprx epower features
		pstat_data = add_to_pstat_data(pstat_data, 'apprx_epower')
		pstat_data = add_to_pstat_data(pstat_data, 'wbinstat_nonoise')


		for reduce_var in [1, 10, 50]:

			# Wstats
			Wstats = knockadapt.adaptive.weighted_binomial_feature_stat(
				Ws, gsizes, eps=0.05, delta=0.01, reduce_var=reduce_var,
				num_partitions=num_partitions
			)
			ns_results[f'wstat_{reduce_var}'] = Wstats
			pstat_data = add_to_pstat_data(pstat_data, f'wstat_{reduce_var}')

	else:
		for feature in partition_stats:
			pstat_data = add_to_pstat_data(pstat_data, feature)
	
	# Step 7: Combine
	reg_data = pd.concat(
		[oracle_data, nonsplit_data, rec_data, baseline_data, pstat_data], 
		axis=0, 
		ignore_index=True,
		sort=True
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

	if modify_oracle:
		return reg_data, oracle_results

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
		print("Note regular data does exist!")
		results = pd.read_csv(csv_path)
		results = results.drop('Unnamed: 0', axis = 'columns')
	else:
		results = None
		print("Note regular data does not exist")

	# Oracle data	
	oracle_path = path + '_oracle.csv'
	if os.path.exists(oracle_path):
		print("Note oracle data does exist!")
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
	elif n_vals.shape[0] != 1 or results is None:
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
		results, oracle_results = construct_reg_from_oracle(oracle_results, x_axis=x_axis, q=q)
	# print("========================================= MEAN =========================================")
	# print(oracle_results.groupby(['n', 'num_groups'])['wbinstat_nonoise'].mean())
	# print("========================================= STD =========================================")
	# print(oracle_results.groupby(['n', 'num_groups'])['wbinstat_nonoise'].std())


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
	fact_x_axis = 'factor(' + x_axis + ')'
	print(fact_x_axis)
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
		print('------ \n ------ ONLY CONSIDERING NON-SPLIT ORACLE (vanilla) ------- \n ----------')

		oracle_results = oracle_results.loc[oracle_results['oracle_type'] == 'oracle']
		#raise ValueError("Too many oracle types present")

	# Round and string-ify cutoffs
	all_cutoff_counts = []
	partition_stats = ['apprx_epower', 'wbinstat_nonoise', 'epower', 'power']
	for partition_stat in partition_stats:
		oracle_results['num_groups'] = np.round(oracle_results['num_groups'], 1)
		g1 = (
			ggplot(oracle_results, aes(
				x='factor(num_groups)', y=partition_stat, 
				fill='factor(num_groups)', group='factor(num_groups)')
			)
			+ geom_boxplot(position='dodge')
			+ facet_grid(f'feature_fn~{x_axis}')
		)
		new_path = path + f'/{partition_stat}_dists.JPG'
		g1.save(new_path)

		# Distribution of best cutoff
		chosen_cutoff_ids = oracle_results.groupby(
			['sample', x_axis, 'feature_fn', 'oracle_type']
		)[partition_stat].idxmax().values
		chosen_cutoffs = oracle_results.loc[chosen_cutoff_ids]

		# Density for each setting
		num_samples = oracle_results['sample'].max() + 1
		cutoff_counts = chosen_cutoffs.groupby(
			[x_axis, 'feature_fn', 'oracle_type'], 
		)['num_groups'].value_counts(normalize=True).unstack().fillna(0).reset_index()
		cutoff_counts = pd.melt(
			cutoff_counts,
			id_vars=[x_axis, 'feature_fn', 'oracle_type'],
			var_name='num_groups',
			value_name='proportion'
		)
		cutoff_counts['partition_stat'] = partition_stat
		all_cutoff_counts.append(cutoff_counts)

		g1 = (
			ggplot(
				cutoff_counts, aes(
				x='num_groups', y='proportion', fill='factor(num_groups)',
				)
			)
			+ geom_col(position='dodge')
			+ facet_grid(f'feature_fn~{x_axis}')
		)
		new_path = path + f'/{partition_stat}_selection_dists.JPG'
		g1.save(new_path)

	# Possibly can plot them all at once
	if oracle_results['feature_fn'].unique().shape[0] == 1:

		# Power/epower distributions
		id_vars = [c for c in oracle_results.columns if c not in partition_stats]
		dist_results = pd.melt(
			oracle_results, 
			id_vars=id_vars,
			var_name='partition_stat',
			value_name='statistic'
		)
		g1 = (
			ggplot(dist_results, aes(
				x='factor(num_groups)', y='statistic', 
				fill='factor(num_groups)', group='factor(num_groups)')
			)
			+ geom_boxplot(position='dodge')
			+ facet_grid(f'partition_stat~{x_axis}', scales='free_y')
		)
		new_path = path + f'/all_dists.JPG'
		g1.save(new_path)





		# Selection distributions
		all_cutoff_counts = pd.concat(all_cutoff_counts, axis=0)
		g1 = (
			ggplot(
				all_cutoff_counts, aes(
				x='num_groups', y='proportion', fill='factor(num_groups)',
				)
			)
			+ geom_col(position='dodge')
			+ facet_grid(f'partition_stat~{x_axis}')
		)
		new_path = path + f'/all_selection_dists.JPG'
		g1.save(new_path)

	# Correlation between cutoffs - start by aligning 
	# similar samples 
	var = 'fdp' # epower or fdp
	all_groupnums = oracle_results['num_groups'].unique()
	merge_on = ['link_method', 'feature_fn', 'sample', x_axis, 'oracle_type']
	all_subsets = []
	for num_groups in all_groupnums:

		# Aggregate subsets
		subset = oracle_results.loc[oracle_results['num_groups'] == num_groups]
		subset = subset.rename(columns = {var:f'{var}_{num_groups}'})
		subset = subset[merge_on + [f'{var}_{num_groups}']]
		# Append
		all_subsets.append(subset)

	# Merge
	aligned_data = functools.reduce(
		lambda left,right: pd.merge(left,right,on=merge_on, how='outer'), 
		all_subsets, 
	)
	corr_columns = [c for c in aligned_data.columns if var in c]
	group_vars = ['link_method', 'feature_fn', x_axis, 'oracle_type']
	corrs = aligned_data.groupby(group_vars)[corr_columns].corr().reset_index()
	corrs = corrs.rename(columns={'level_4':'left_group_num'})
	corrs = pd.melt(
		corrs,
		id_vars=group_vars+['left_group_num'],
		var_name='right_group_num',
		value_name='correlation'
	)
	corrs['left_group_num'] = corrs['left_group_num'].str.replace(f'{var}_', '').astype(int)
	corrs['right_group_num'] = corrs['right_group_num'].str.replace(f'{var}_', '').astype(int)
	corrs['correlation'] = np.around(corrs['correlation'], 3)


	# Plot correlations 
	corrplot = (
		ggplot(corrs, aes(
			x='factor(left_group_num)',
			y='factor(right_group_num)', 
			fill='correlation', 
			color='correlation'
		))
		+ geom_tile()
		+ geom_text(aes(label='correlation'), color='white')
		+ facet_grid(f'{x_axis}~feature_fn')

	)
	new_path = path + f'/{var}_corrplot.JPG' 
	corrplot.save(new_path)

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
