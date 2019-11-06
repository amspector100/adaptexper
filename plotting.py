import sys
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
from plotnine import *
warnings.simplefilter("always")

def plot_measurement_type(melted_results, 
						  meas_type = 'power',
						  fname = None,
						  yintercept = None):
	""" Plotting and saving function """

	results = melted_results.loc[melted_results['measurement'] == meas_type]
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
	q_path = path + '_q.txt'
	with open(q_path, 'r') as thefile:
		q = thefile.read()
	q = float(q)
	results = pd.read_csv(csv_path)

	# Plot power
	power_path = path + '_power.SVG'
	power_results = results.loc[results['measurement'] == 'power']
	power_results = power_results.rename(columns = {'value':'power'})
	g2 = (
		ggplot(power_results, aes(
			x = 'n', y = 'power', color = 'split_type')
		)
		+ stat_summary(geom = 'line')
		+ stat_summary(geom = "errorbar", fun_data = 'mean_cl_boot')
		+ stat_summary(aes(shape = 'split_type'), geom = 'point', size = 2.5)
		+ facet_grid('feature_fn~link_method')
		+ labs(title = path)
	)
	g2.save(power_path)

	# Plot FDR
	fdr_path = path + '_fdr.SVG'
	fdr_results = results.loc[results['measurement'] == 'fdr']
	fdr_results = fdr_results.rename(columns = {'value':'fdr'})
	hline = geom_hline(
		aes(yintercept = q), linetype="dashed", color = "red"
	)
	g2 = (
		ggplot(fdr_results, aes(
			x = 'n', y = 'fdr', color = 'split_type')
		)
		+ stat_summary(geom = 'line')
		+ stat_summary(geom = "errorbar", fun_data = 'mean_cl_boot')
		+ stat_summary(aes(shape = 'split_type'), geom = 'point', size = 2.5)
		+ facet_grid('feature_fn~link_method')
		+ hline
		+ labs(title = path)
	)
	g2.save(fdr_path)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Please specify the path to create plots for')
	else:
		sys.exit(plot_n_curve(sys.argv[1]))