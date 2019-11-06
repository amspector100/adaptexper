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

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Please specify the path of the csv to create plots for')
	else:
		sys.exit(plot_csv(sys.argv[1]))