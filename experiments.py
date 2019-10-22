import numpy as np
from scipy import stats
import knockadapt
from knockadapt.knockoff_stats import calc_nongroup_LSM

import scipy.cluster.hierarchy as hierarchy

import time
import pandas as pd
import matplotlib.pyplot as plt


# Note full avg power is over all groupings
OUTPUT_COLUMNS = [
    'actual_power', 'actual_empirical_power', 'actual_fdr',
    'nonsplit_power', 'nonsplit_empirical_power', 'nonsplit_fdr',
    'oracle_power', 'oracle_empirical_power', 'oracle_fdr',
    'full_avg_power'
]
OUTPUT_COLUMNS_V2 = [
    'link_method', 'feature_fn', 'sample',
    'actual_power', 'actual_empirical_power', 
    'actual_expected_empirical_power', 'actual_fdr',
    'nonsplit_power', 'nonsplit_empirical_power', 'nonsplit_fdr',
    'oracle_power', 'oracle_empirical_power', 'oracle_fdr',
]

# Plotting functions ------------------------------------------
def add_extra_x_axis(ax, cutoffs, Ms): 
    """ This function has global effects, beware!"""
    
    # Add group size ticks for ax1
    new_ax = ax.twiny()
    old_ticks = ax.get_xticks()
    new_ticks = []
    for i, x in enumerate(cutoffs):
        new_ticks.append(Ms[i])
        
    new_ax.set_xticks(old_ticks)
    new_ax.set_xbound(ax.get_xbound())
    new_ax.set_xticklabels(new_ticks)

def plot_powers(x, hat_powers, fdps, powers, Ms,
                x_label = 'Correlation cutoff'):
    """ xs are cutoffs """
    
    # Np-ify
    x = np.array(x)
    hat_powers = np.array(hat_powers)
    fdps = np.array(fdps)
    powers = np.array(powers)
    
    # Plot!
    fig, (ax0, ax1) = plt.subplots(ncols = 2, figsize = (16, 6))
    ax0.plot(x, hat_powers, color = 'darkgreen', 
             label = 'Empirical powers')
    ax0.scatter(x, hat_powers, color = 'darkgreen', label = None)
    ax0.plot(x, powers, color = 'navy', 
             label = 'True powers')
    ax0.scatter(x, powers, color = 'navy', label = None)
    ax0.set_xticks(x)
    ax0.legend()
    ax0.set(title = 'Group Powers', xlabel = x_label, ylabel = 'Power')
    add_extra_x_axis(ax0, x, Ms)
    
    # FDRs
    ax1.plot(x, fdps, color = 'black', label = 'Observed')
    ax1.scatter(x, fdps, color = 'black', label = None)
    ax1.axhline(y = q, linestyle ='dashed', color = 'red', label = 'Targetted')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.set(title = 'FDRs', xlabel = x_label, ylabel = 'FDR')
    add_extra_x_axis(ax1, x, Ms)

    # Plot title
    fig.suptitle(f'Group adaptive knockoffs for n = {n}, p = {p}, coefficient mag = {coeff_size}')
    plt.show()


def test_proposed_methods(n = 100,
                          p = 50, 
                          q = 0.25, 
                          num_graphs = 20,
                          link_method = 'complete',
                          split = True,
                          sample_kwargs = {'coeff_size':10,},
                          knockoff_methods = {'default':{}},
                          seed = 110,
                          plot = False,
                          ):
    """ Find expected power of a proposed method with a specific
    method of data sampling / graph generation 

    :param knockoff_methods: A dictionary of method names
    which map to kwargs to pass to the knockoff constructor."""


    # Possibly make reproducible
    if seed is not None:
        np.random.seed(seed)

    # Initialize data structure to hold outputs
    method_names = list(knockoff_methods.keys())
    indexes = np.arange(0, num_graphs, 1)
    data = {
        method:pd.DataFrame(columns = OUTPUT_COLUMNS) for method in method_names
    }

    # Timing
    time0 = time.time()

    for j in range(num_graphs):

        # Sample data, create correlation tree, possibly plot
        X, y, beta, Q, corr_matrix = knockadapt.graphs.sample_data(
           n = n, p = p, **sample_kwargs
        )
        link = knockadapt.graphs.create_correlation_tree(
            corr_matrix, method = link_method
        )
        if plot:
            knockadapt.graphs.plot_dendrogram(
                link, title = link_method + ' Hierarchical Clustering'
            )

        for method_name in method_names:
            print(f'Beginning trial {j}, method {method_name}, time is {time.time() - time0}')
            knockoff_kwargs = knockoff_methods[method_name]

            # Calculate oracle powers
            reduction = int(np.floor(p/5))
            cutoffs, hat_powers, fdps, powers, Ms = knockadapt.adaptive.select_highest_power(
                X, y, corr_matrix, link, non_nulls = beta, q = q, reduction = reduction,
                **knockoff_kwargs, 
            )

            # Find oracle power, empirical power, fdr
            full_avg_power = np.mean(np.array(powers))
            selection = np.argmax(np.array(powers))
            oracle_power = powers[selection]
            oracle_empirical_power = hat_powers[selection]
            oracle_fdr = fdps[selection]

            # Find non-sample-split powers
            selection = np.argmax(np.array(hat_powers))
            nonsplit_power = powers[selection]
            nonsplit_empirical_power = hat_powers[selection]
            nonsplit_fdr = fdps[selection]

            # Calculate what we'd actually see with actual powers
            if split:

                # Split in half and "train"
                trainX = X[0:int(n/2)]
                trainy = y[0:int(n/2)]
                cutoffs, hat_powers, fdps, powers, Ms = knockadapt.adaptive.select_highest_power(
                    trainX, trainy, corr_matrix, link, non_nulls = beta, q = q, reduction = reduction,
                    **knockoff_kwargs, 
                )

                # Pick our best grouping/expected power
                actual_empirical_power = max(hat_powers)
                selected_cutoff = cutoffs[np.argmax(np.array(hat_powers))]
                selected_grouping = hierarchy.fcluster(
                    link, selected_cutoff, criterion = "distance"
                )

                # Now test to see what discoveries we find
                testX = X[int(n/2):]
                testy = y[int(n/2):]
                fdps, powers, hat_powers = knockadapt.adaptive.evaluate_grouping(
                    testX, testy, corr_matrix, selected_grouping, non_nulls = beta, q =q,
                    **knockoff_kwargs,
                )
                actual_fdr = np.array(fdps).mean()
                actual_power = np.array(powers).mean()

            # Add to dataframe
            method_results = [
                actual_power, actual_empirical_power, actual_fdr,
                nonsplit_power, nonsplit_empirical_power, nonsplit_fdr,
                oracle_power, oracle_empirical_power, oracle_fdr,
                full_avg_power
            ]
            data[method_name].loc[j] = pd.Series(method_results, index = OUTPUT_COLUMNS)

    # Coerce data into proper format
    all_data = pd.DataFrame(columns = OUTPUT_COLUMNS + ['methodname'])
    for methodname in data:
        data[methodname]['methodname'] = methodname
        all_data = pd.concat([all_data, data[methodname]])
    all_data['trial'] = all_data.index
    all_data.reset_index(inplace = True)

    return all_data

def compare_methods(
                    corr_matrix,
                    beta,
                    Q = None,
                    n = 500, 
                    q = 0.25, 
                    num_data_samples = 10,
                    link_methods = ['complete', 'single', 'average'],
                    split = True,
                    sample_kwargs = {'coeff_size':10,},
                    feature_fns = {'LSM':calc_nongroup_LSM},
                    S_kwargs = {'objective':'norm', 'norm_type':'fro'},
                    copies = 1,
                    seed = 110,
                    reduction = 10,
                    ):
    """ 
    In the future, could add an 'S_methods' optional arg
    """

    # Possibly make reproducible, also time
    time0 = time.time()
    if seed is not None:
        np.random.seed(seed)

    # Get p, Q
    p = corr_matrix.shape[0]
    if Q is not None:
        Q = knockadapt.utilities.chol2inv(corr_matrix)

    # Sample data for the first time, create links
    X, y, _, _, _ = knockadapt.graphs.sample_data(
       n = n, p = p, corr_matrix = corr_matrix, 
       Q = Q, beta = beta, **sample_kwargs
    )

    # Create links, groups, cutoffs
    links = {
        link_method:knockadapt.graphs.create_correlation_tree(
            corr_matrix, method = link_method
        ) for link_method in link_methods
    }

    # List of list storing cutoffs
    all_cutoffs = {
        link_method:links[link_method][:, 2][::reduction] for link_method in link_methods
    }

    # Dictionary of dictionaries (link by cutoff) which stores group sizes
    all_Ms = {}
    # Dictionary of dictionaries (link by cutoff) which stores groupings
    all_groups = {}
    for link_method in link_methods:
        # Graph cutoffs, links
        cutoffs = all_cutoffs[link_method]
        link = links[link_method]

        # Create groups for each cutoff
        link_groups = {}
        Ms = {}
        for cutoff in cutoffs:
            groups = hierarchy.fcluster(link, cutoff, criterion = "distance")
            link_groups[cutoff] = groups
            Ms[cutoff] = np.unique(groups).shape[0]

        # Add smaller dictionaries to parent dictionaries
        all_groups[link_method] = link_groups
        all_Ms[link_method] = Ms

    # Create S matrices: dictionary of dictionaries (link by cutoff)
    S_matrixes = {link_method:{} for link_method in link_methods}
    for link_method in link_methods:

        # Retrive groups/cutoffs for this link method
        link_method_groups = all_groups[link_method]
        cutoffs = all_cutoffs[link_method]

        # Progress report
        print(f'Generating S matrices for {link_method} now, time is {time.time() - time0}')

        # Add S matrixes
        for cutoff in cutoffs:
            groups = link_method_groups[cutoff]
            _, S_group = knockadapt.knockoffs.group_gaussian_knockoffs(
                X = X, Sigma = corr_matrix, groups = groups,
                invSigma = Q, return_S = True, **S_kwargs
            )
            S_matrixes[link_method][cutoff] = S_group

    # Construct oracle (curse of dimensionality applies here)
    feature_methods = [fname for fname in feature_fns]
    columns = ['sample', 'cutoff', 'feature_fn', 'link_method', 'power', 'fdp']
    oracle_results = pd.DataFrame(columns = columns)

    print("Picking the best oracle!")
    for j in range(num_data_samples):

        print(f'At data sample {j} for oracle, time is {time.time() - time0}')

        # Create X and y
        X, y, beta2, Q2, corr_matrix2 = knockadapt.graphs.sample_data(
           n = n, p = p, corr_matrix = corr_matrix, Q = Q, beta = beta,
           **sample_kwargs
        )

        # Sanity check
        if np.abs(beta2 - beta).sum() != 0:
            raise ValueError('Uh oh, DGP is being changed! (beta)')
        if np.abs(corr_matrix2 - corr_matrix).sum() != 0:
            raise ValueError('Uh oh, DGP is being changed! (corr_matrix)')
        if np.abs(Q2 - Q).sum() != 0:
            raise ValueError('Uh oh, DGP is being changed! (Q)')

        # Iterate through link methods and feature methods
        for link_method in link_methods:
            for feature_method in feature_fns:

                # Right feature statistic generator
                feature_stat_fn = feature_fns[feature_method]

                # Run knockoffs for each cutoff
                for cutoff in all_cutoffs[link_method]:

                    # Get the group and S matrix
                    groups = all_groups[link_method][cutoff]
                    S = S_matrixes[link_method][cutoff]

                    fdps, powers, hat_powers = knockadapt.adaptive.evaluate_grouping(
                        X = X, y = y, corr_matrix = corr_matrix, groups = groups, q = q,
                        non_nulls = beta, S = S, copies = copies, verbose = False,
                        feature_stat_fn = feature_stat_fn
                    )

                    # Add power
                    to_add = pd.DataFrame(
                        columns = columns,
                        data = [[j, cutoff, feature_method, link_method, powers.mean(), fdps.mean()]]
                    )
                    oracle_results = oracle_results.append(to_add)

    # Pick best cutoffs based on mean power
    mean_powers = oracle_results.groupby(
        ['feature_fn', 'link_method', 'cutoff']
    )['power'].mean()
    oracle_cutoffs = mean_powers.unstack().idxmax(1).unstack()

    print('Finished creating oracle: comparing methods now')

    # Initialize output, begin loop
    output_df = pd.DataFrame(columns = OUTPUT_COLUMNS_V2)
    for j in range(num_data_samples):

        print(f'At data sample {j} for methods, time is {time.time() - time0}')

        # Create X and y
        X, y, beta2, Q2, corr_matrix2 = knockadapt.graphs.sample_data(
           n = n, p = p, corr_matrix = corr_matrix, Q = Q, beta = beta,
           **sample_kwargs
        )

        # Iterate through link methods and feature methods
        for link_method in link_methods:
            for feature_method in feature_fns:

                # Right feature statistic generator
                feature_stat_fn = feature_fns[feature_method]
                link = links[link_method]

                # Oracle cutoff
                oracle_cutoff = oracle_cutoffs.loc[feature_method, link_method]
                groups = all_groups[link_method][oracle_cutoff]
                S = S_matrixes[link_method][oracle_cutoff]

                # Calculate oracle power --------
                o_fdps, o_powers, o_hat_powers = knockadapt.adaptive.evaluate_grouping(
                    X = X, y = y, corr_matrix = corr_matrix, groups = groups, q = q,
                    non_nulls = beta, S = S, copies = copies, verbose = False,
                    feature_stat_fn = feature_stat_fn
                )
                oracle_fdr = o_fdps.mean()
                oracle_power = o_powers.mean()
                oracle_empirical_power = o_hat_powers.mean()

                # Find cutoffs, list of groups
                link_cutoffs = all_cutoffs[link_method]
                link_groups = all_groups[link_method]
                link_S_matrices = S_matrixes[link_method]

                # Calculate adaptive power for nonsplit method -----
                _, ns_hat_powers, ns_fdps, ns_powers, _ = knockadapt.adaptive.select_highest_power(
                    X = X, y = y, corr_matrix = corr_matrix, link = link,
                    q = q, cutoffs = link_cutoffs, non_nulls = beta, 
                    reduction = reduction,
                    S_matrices = link_S_matrices,
                    copies = copies, verbose = False, 
                    feature_stat_fn = feature_stat_fn,
                )

                # Find non-sample-split powers
                nonsplit_selection = np.argmax(np.array(ns_hat_powers))
                nonsplit_power = ns_powers[nonsplit_selection]
                nonsplit_empirical_power = ns_hat_powers[nonsplit_selection]
                nonsplit_fdr = ns_fdps[nonsplit_selection]

                # Repeat but for sample-splitting method ----------

                # Split in half and "train"
                trainX = X[0:int(n/2)]
                trainy = y[0:int(n/2)]
                _, spl_hat_powers, spl_fdps, spl_powers, _ = knockadapt.adaptive.select_highest_power(
                    X = trainX, y = trainy, corr_matrix = corr_matrix, link = link,
                    q = q, cutoffs = link_cutoffs, non_nulls = beta, 
                    reduction = reduction,
                    S_matrices = link_S_matrices,
                    copies = copies, verbose = False, 
                    feature_stat_fn = feature_stat_fn,
                )

                # Pick our best grouping/expected power
                expected_empirical_power = max(spl_hat_powers)
                selected_cutoff = link_cutoffs[np.argmax(np.array(hat_powers))]
                selected_grouping = link_groups[selected_cutoff]

                # Now test to see what discoveries we find
                testX = X[int(n/2):]
                testy = y[int(n/2):]
                spl_fdps, spl_powers, spl_hat_powers = knockadapt.adaptive.evaluate_grouping(
                    X = testX, y = testy, corr_matrix = corr_matrix, groups = groups, q = q, 
                    non_nulls = beta, S = S, copies = copies, verbose = False,
                    feature_stat_fn = feature_stat_fn
                )
                actual_fdr = spl_fdps.mean()
                actual_power = spl_powers.mean()
                actual_empirical_power = spl_hat_powers.mean()

                # Add to dataframe
                method_results = [
                    link_method, feature_method, j,
                    actual_power, actual_empirical_power, 
                    expected_empirical_power, actual_fdr,
                    nonsplit_power, nonsplit_empirical_power, nonsplit_fdr,
                    oracle_power, oracle_empirical_power, oracle_fdr,
                ]
                output_df = output_df.append(
                    pd.DataFrame([method_results], columns = OUTPUT_COLUMNS_V2),
                    ignore_index = True
                )

    # Reshape results
    melted_results = pd.melt(output_df, 
                             id_vars = ['link_method', 'feature_fn', 'sample'])
    melted_results['measurement'] = melted_results['variable'].apply(
        lambda x: ('_').join(x.split('_')[1:])
    )
    melted_results['split_type'] = melted_results['variable'].apply(
        lambda x: x.split('_')[0]
    )
    id_vars = ['link_method', 'feature_fn', 'split_type', 'measurement']

    return melted_results, oracle_results, S_matrixes



        

