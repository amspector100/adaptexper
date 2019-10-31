import warnings
warnings.filterwarnings("ignore")
from plotnine import *
warnings.simplefilter("always")

def plot_measurement_type(melted_results, 
                          meas_type = 'power',
                          fname = None,
                          hline = None):
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
    if hline is not None:
        g1 = g1 + hline
    
    if fname is not None:
        g1 = g1 + labs(title = fname)
        fname1 = fname + '_' + meas_type + '_v1.SVG'
        g1.save(fname1)
    print(g1)
    
    g2 = (
        ggplot(results, aes(
            x = 'feature_fn', y = meas_type, fill = 'split_type')
        )
        + stat_summary(geom = 'col', position=position_dodge(width=0.9))
        + stat_summary(geom = "errorbar", fun_data = 'mean_cl_boot',
                        position=position_dodge(width=0.9))
        + facet_grid('~link_method')
    )
    
    if hline is not None:
        g2 = g2 + hline
        
    if fname is not None:
        g2 = g2 + labs(title = fname)
        fname2 = fname + '_' + meas_type + '_v2.SVG'
        g2.save(fname2)        
    print(g2)
