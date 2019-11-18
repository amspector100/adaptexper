import sys
import numpy as np
from scipy import stats
import knockadapt
from knockadapt.knockoff_stats import calc_nongroup_LSM
import scipy.cluster.hierarchy as hierarchy

import time
import pandas as pd
import matplotlib.pyplot as plt

from smatrices import cache_S_matrix, load_S_matrix

from multiprocessing import Pool
from functools import partial 

# Note full avg power is over all groupings
FINAL_COLUMNS = ['link_method', 'feature_fn', 'sample', 'cutoff', 
                'num_groups', 'variable', 'value', 'measurement',
                'split_type']

ORACLE_COLUMNS = ['sample', 'cutoff', 'feature_fn', 'link_method',
                  'power', 'fdp', 'oracle_type']

### -------------- HELPER FUNCTIONS FOR MULTIPROCESSING -----------------


def compute_S_matrix(S_group, link_method, cutoff,
                    X, corr_matrix, invSigma, groups, 
                    S_kwargs, S_method, p, seed, sample_kwargs):
    """ Helper function for multiprocessing """ 

    # Compute the matrix if it hasn't been loaded
    if S_group is None:

        _, S_group = knockadapt.knockoffs.group_gaussian_knockoffs(
            X = X, Sigma = corr_matrix, groups = groups,
            invSigma = invSigma, return_S = True, **S_kwargs,
            **S_method[1]
        )

        # But save it!
        cache_S_matrix(S_group, 
                       p, seed, 
                       cutoff, 
                       link_method, 
                       sample_kwargs)

    return S_group, link_method, cutoff


def eval_oracles(j, n, p, q, X, y, corr_matrix, Q, beta, sample_kwargs,
                 link_methods, feature_fns, feature_fn_kwargs,
                 all_cutoffs, all_groups, S_matrixes, time0, copies, 
                 compute_split_oracles = True):
    """ Helper function for multiprocessing: evaluates the oracles, returns
    a list of rows to add to the oracle results dataframe.
    Sorry for all the arguments, it's unavoidable with the multiprocessing package
    (This has to be globally defined or it's not pickleable)

    Note: j is the sample number 
    If compute_split_oracles is False, will only do one oracle"""

    sys.stdout.write(f'At data sample {j} for oracle, time is {time.time() - time0}\n')

    # Create X and y - make this random so different
    # processes really have different values. But save random state
    # and restore it for later
    st0 = np.random.get_state()
    np.random.seed()
    X, y, beta2, Q2, corr_matrix2 = knockadapt.graphs.sample_data(
       n = n, p = p, corr_matrix = corr_matrix, Q = Q, beta = beta,
       **sample_kwargs
    )
    # Restore random state
    np.random.set_state(st0)

    # Sanity check
    if np.abs(beta2 - beta).sum() != 0:
        raise ValueError('Uh oh, DGP is being changed! (beta)')
    if np.abs(corr_matrix2 - corr_matrix).sum() != 0:
        raise ValueError('Uh oh, DGP is being changed! (corr_matrix)')
    if np.abs(Q2 - Q).sum() != 0:
        raise ValueError('Uh oh, DGP is being changed! (Q)')

    # Iterate through link methods and feature methods
    # to create inputs for multiprocessed S matrix
    outputs_to_add = []
    for link_method in link_methods:
        for feature_method in feature_fns:

            # Right feature statistic generator
            feature_stat_fn = feature_fns[feature_method]
            feature_stat_kwargs = feature_fn_kwargs[feature_method]

            # Run knockoffs for each cutoff
            for cutoff in all_cutoffs[link_method]:

                # Get the group and S matrix
                groups = all_groups[link_method][cutoff]
                S = S_matrixes[link_method][cutoff]

                # Oracle for full data
                fdps, powers, hat_powers = knockadapt.adaptive.evaluate_grouping(
                    X = X, y = y, corr_matrix = corr_matrix, groups = groups, q = q,
                    non_nulls = beta, S = S, copies = copies, verbose = False,
                    feature_stat_fn = feature_stat_fn, feature_stat_kwargs = feature_stat_kwargs,
                )

                # Add power to regular oracle
                to_add = pd.DataFrame(
                    columns = ORACLE_COLUMNS,
                    data = [[j, cutoff, feature_method, link_method, 
                            powers.mean(), fdps.mean(), 'oracle']]
                )
                outputs_to_add.append(to_add)

                if compute_split_oracles:

                    # Add power to non-recycling split oracle

                    # Oracle for half data - does NOT use sample recycling
                    half_fdps, half_powers, half_hat_powers = knockadapt.adaptive.evaluate_grouping(
                        X = X[0:int(n/2)], y = y[0:int(n/2)], 
                        corr_matrix = corr_matrix, groups = groups, q = q,
                        non_nulls = beta, S = S, copies = copies, verbose = False,
                        feature_stat_fn = feature_stat_fn, feature_stat_kwargs = feature_stat_kwargs,
                    )

                    # Add to outputs
                    half_to_add = pd.DataFrame(
                        columns = ORACLE_COLUMNS,
                        data = [[j, cutoff, feature_method, link_method, 
                                 half_powers.mean(), half_fdps.mean(), 'split_oracle']]
                    )

                    outputs_to_add.append(half_to_add)


                    # Oracle for half data which DOES use sample recycling
                    rec_fdps, rec_powers, rec_hat_powers = knockadapt.adaptive.evaluate_grouping(
                        X = X[0:int(n/2)], y = y[0:int(n/2)], 
                        corr_matrix = corr_matrix, groups = groups, q = q,
                        non_nulls = beta, S = S, copies = copies, verbose = False,
                        feature_stat_fn = feature_stat_fn, feature_stat_kwargs = feature_stat_kwargs,
                    )

                    # Add power to recycling split oracle
                    rec_to_add = pd.DataFrame(
                        columns = ORACLE_COLUMNS,
                        data = [[j, cutoff, feature_method, link_method, 
                                rec_powers.mean(), rec_fdps.mean(), 'rec_oracle']]
                    )


                    outputs_to_add.append(rec_to_add)

    return outputs_to_add

def to_add_to_final_df(fdr,
                       power,
                       epower,
                       cutoff,
                       num_groups,
                       link_method,
                       feature_fn,
                       sample,
                       split_type,
                       ):
    """ Helper function for compare_methods, constructs DFs 
    to add to the final output. Epower stands for empirical power"""

    # Create melted lists of the form:
    # link_method, feature_fn, sample, cutoff, num_groups variable, value, measurement, split_type
    base_list = [link_method, feature_fn, sample, cutoff, num_groups]
    power_list = base_list + ['power', power, split_type + '_power', split_type]
    epower_list = base_list + [
    'empirical_power', epower, split_type + '_empirical_power', split_type
    ]
    fdr_list = base_list + ['fdr', fdr, split_type + '_fdr', split_type]

    # Add to dataframe
    output_list = []
    for list_to_add in [power_list, fdr_list, epower_list]:
        to_add = pd.DataFrame(
            columns = FINAL_COLUMNS,
            data = [list_to_add]
        )
        output_list.append(to_add)
    return output_list

def one_sample_comparison(j, n, p, q, X, y, corr_matrix, Q, beta, sample_kwargs,
                          reduction, links, link_methods, feature_fns, feature_fn_kwargs,
                          all_cutoffs, all_groups, S_matrixes, time0,
                          copies, all_oracle_cutoffs):
    """ Compares methods with one sample - this is a helper function
    to pass to the multiprocessing package later. As before, j is sample number."""

    sys.stdout.write(f'At sample labeled {j} for methods, time is {time.time() - time0}\n')

    # Create X and y - try to really randomize this so it's
    # different between different processes.
    st0 = np.random.get_state()
    np.random.seed()
    X, y, _, _, _ = knockadapt.graphs.sample_data(
       n = n, p = p, corr_matrix = corr_matrix, Q = Q, beta = beta,
       **sample_kwargs
    )
    # Restore random state
    np.random.set_state(st0)

    # Initialize final output
    final_output = []

    # Iterate through link methods and feature methods
    for link_method in link_methods:
        for feature_method in feature_fns:

            # Right feature statistic generator
            feature_stat_fn = feature_fns[feature_method]
            feature_stat_kwargs = feature_fn_kwargs[feature_method]
            link = links[link_method]

            # Calcualte oracle powers, fdrs, etc ---------------
            trainX = X[0:int(n/2)]
            trainy = y[0:int(n/2)]
            for oracle_type in all_oracle_cutoffs:

                # Cutoff, links, groups
                oracle_cutoffs = all_oracle_cutoffs[oracle_type]
                oracle_cutoff = oracle_cutoffs.loc[feature_method, link_method]
                
                # In the case of the global null, sometimes there's no cutoff
                if np.isnan(oracle_cutoff):
                    oracle_cutoff = all_cutoffs[link_method][0]

                groups = all_groups[link_method][oracle_cutoff]
                num_groups = np.unique(groups).shape[0]
                S = S_matrixes[link_method][oracle_cutoff]

                # One of the oracles uses split data
                if oracle_type == 'split_oracle':
                    X_to_use = trainX
                    y_to_use = trainy
                else:
                    X_to_use = X
                    y_to_use = y
                
                # One of the oracles recycles
                if oracle_type == 'rec_oracle':
                    recycle_up_to = int(n/2)
                else:
                    recycle_up_to = None

                # Calculate oracle power --------
                o_fdps, o_powers, o_hat_powers = knockadapt.adaptive.evaluate_grouping(
                    X = X_to_use, y = y_to_use, corr_matrix = corr_matrix, 
                    groups = groups, q = q, non_nulls = beta,
                     S = S, copies = copies, verbose = False,
                    feature_stat_fn = feature_stat_fn,
                    feature_stat_kwargs = feature_stat_kwargs,
                    recycle_up_to = recycle_up_to
                )
                oracle_fdr = o_fdps.mean()
                oracle_power = o_powers.mean()
                oracle_empirical_power = o_hat_powers.mean()

                # Add to output
                to_add = to_add_to_final_df(
                    fdr = oracle_fdr, power = oracle_power,
                    epower = oracle_empirical_power,
                    cutoff = oracle_cutoff, num_groups = num_groups,
                    link_method = link_method, 
                    feature_fn = feature_method, 
                    sample = j, split_type = oracle_type
                )
                final_output.extend(to_add)


            # Find cutoffs, list of groups for other methods -----
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
                feature_stat_kwargs = feature_stat_kwargs,
            )

            # Find non-sample-split powers
            nonsplit_selection = np.argmax(np.array(ns_hat_powers))
            nonsplit_cutoff = link_cutoffs[nonsplit_selection]
            nonsplit_num_groups = np.unique(link_groups[nonsplit_cutoff]).shape[0]
            nonsplit_power = ns_powers[nonsplit_selection]
            nonsplit_empirical_power = ns_hat_powers[nonsplit_selection]
            nonsplit_fdr = ns_fdps[nonsplit_selection]

            # Add to output
            to_add = to_add_to_final_df(
                fdr = nonsplit_fdr, power = nonsplit_power,
                epower = nonsplit_empirical_power,
                cutoff = nonsplit_cutoff,
                num_groups = nonsplit_num_groups, 
                link_method = link_method,
                feature_fn = feature_method,
                sample = j, split_type = 'nonsplit'
            )
            final_output.append(to_add)

            # Also, add the baseline to the output
            if link_cutoffs[0] != 0:
                raise ValueError(f'Expected cutoffs to start with 0, instead were {link_cutoffs}')
            baseline_cutoff = 0
            baseline_num_groups = np.unique(link_groups[baseline_cutoff]).shape[0]
            baseline_power = ns_powers[0]
            baseline_empirical_power = ns_powers[0]
            baseline_fdr = ns_fdps[0]

            # Add baseline to output
            to_add = to_add_to_final_df(
                fdr = baseline_fdr, power = baseline_power,
                epower = baseline_empirical_power,
                cutoff = baseline_cutoff,
                num_groups = baseline_num_groups, 
                link_method = link_method,
                feature_fn = feature_method,
                sample = j, split_type = 'baseline'
            )
            final_output.append(to_add)
            
            # Repeat but for sample-splitting method ----------

            # Split in half and "train"
            _, spl_hat_powers, spl_fdps, spl_powers, _ = knockadapt.adaptive.select_highest_power(
                X = trainX, y = trainy, corr_matrix = corr_matrix, link = link,
                q = q, cutoffs = link_cutoffs, non_nulls = beta, 
                reduction = reduction,
                S_matrices = link_S_matrices,
                copies = copies, verbose = False, 
                feature_stat_fn = feature_stat_fn,
                feature_stat_kwargs = feature_stat_kwargs,
            )

            # Pick our best grouping/expected power
            expected_empirical_power = max(spl_hat_powers)
            selected_cutoff = link_cutoffs[np.argmax(np.array(spl_hat_powers))]
            selected_grouping = link_groups[selected_cutoff]
            spl_num_groups = np.unique(selected_grouping).shape[0]

            # Now test to see what discoveries we find
            # This involves knockoff recycling 
            spl_fdps, spl_powers, spl_hat_powers = knockadapt.adaptive.evaluate_grouping(
                X = X, y = y, corr_matrix = corr_matrix, groups = groups, q = q, 
                non_nulls = beta, S = S, copies = copies, verbose = False,
                recycle_up_to = int(n/2),
                feature_stat_fn = feature_stat_fn,
                feature_stat_kwargs = feature_stat_kwargs,
            )
            actual_fdr = spl_fdps.mean()
            actual_power = spl_powers.mean()
            actual_empirical_power = spl_hat_powers.mean()

            # Add to dataframe
            to_add = to_add_to_final_df(
                fdr = actual_fdr, power = actual_power,
                epower = actual_empirical_power,
                cutoff = selected_cutoff, 
                num_groups = spl_num_groups,
                link_method = link_method,
                feature_fn = feature_method,
                sample = j, split_type = 'actual'
            )
            final_output.append(to_add)

    return final_output


def compare_methods(
                    corr_matrix,
                    beta,
                    Q = None,
                    n = 500, 
                    q = 0.25, 
                    num_data_samples = 10,
                    link_methods = ['average'],
                    S_methods = None,
                    split = True,
                    sample_kwargs = {'coeff_size':10,},
                    feature_fns = {'LSM':calc_nongroup_LSM},
                    feature_fn_kwargs = {},
                    S_kwargs = {'objective':'norm', 'norm_type':'fro'},
                    copies = 1,
                    seed = 110,
                    reduction = None,
                    time0 = None,
                    scache_only = False,
                    num_processes = 8,
                    compute_split_oracles = True,
                    ):
    """ 
    S_methods arg optionally allows you to add extra kwargs (e.g. ASDP instead of SDP)
    for each link method. Should be a list of tuples, of the form
    [(methodname, method_kwargs)], and it should be the same length as 
    link_methods.
    scache_only: If True, only compute the S_group matrices, then stop.
    """

    # Possibly make reproducible, also time
    if time0 is None:
        time0 = time.time()
    if seed is not None:
        np.random.seed(seed)

    # Get p, Q, reduction
    p = corr_matrix.shape[0]
    if Q is None:
        Q = knockadapt.utilities.chol2inv(corr_matrix)
    if reduction is None:
        reduction = 10

    # Sample data for the first time, create links
    X, y, _, _, _ = knockadapt.graphs.sample_data(
       n = n, p = p, corr_matrix = corr_matrix, 
       Q = Q, beta = beta, **sample_kwargs
    )

    # Sometimes the link methods are the same because we're also comparing
    # S generation methods (e.g. ASDP vs SDP), so might have to rename them
    link_method_dict = {}
    if S_methods is not None:
        for i in range(len(link_methods)):
            methodname = S_methods[i][0]
            oldname = link_methods[i]
            new_name = methodname + "_" + oldname
            link_methods[i] = new_name
            link_method_dict[new_name] = oldname

    # Create links, groups, cutoffs
    links = {
        link_method:knockadapt.graphs.create_correlation_tree(
            corr_matrix, method = link_method_dict[link_method]
        ) for link_method in link_methods
    }

    # Dictionary storing cutoff lists for each link method
    all_cutoffs = {}
    for link_method in link_methods:
        link = links[link_method]
        # Max size refers to maximum group size
        cutoffs = knockadapt.adaptive.create_cutoffs(
            link = link, reduction = reduction, max_size = 100
        )
        all_cutoffs[link_method] = cutoffs

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
    # This is a bit hacky, but we can associate a different S_method
    # with each link method if we want.
    if S_methods is None:
        S_methods = [{} for _ in link_methods]

    S_matrixes = {link_method:{} for link_method in link_methods}

    # Assemble the list of parameters to pass to the multiprocessing module
    all_arguments = []
    for link_method, S_method in zip(link_methods, S_methods):

        # Retrive groups/cutoffs for this link method
        link_method_groups = all_groups[link_method]
        cutoffs = all_cutoffs[link_method]

        # Progress report
        sys.stdout.write(f'Generating/retreiving S matrices for {link_method} now, time is {time.time() - time0}\n')

        # Add S matrixes
        for cutoff in cutoffs:
            groups = link_method_groups[cutoff]

            # Possibly load from text file
            S_group = load_S_matrix(p, seed, 
                                    cutoff, link_method,
                                    sample_kwargs)

            if S_group is not None:
                S_matrixes[link_method][cutoff] = S_group
            else:
                all_arguments.append(
                    (S_group, link_method, cutoff,
                     X, corr_matrix, Q, groups, 
                     S_kwargs, S_method, p, seed,
                     sample_kwargs)
                )

    # Pass to multiprocessor
    if num_processes == 1:
        all_S_outputs = []
        for arguments in all_arguments:
            all_S_outputs.append(compute_S_matrix(*arguments))
    else:
        with Pool(num_processes) as thepool:
            all_S_outputs = thepool.starmap(
                compute_S_matrix, all_arguments
            )

    for (S_group, link_method, cutoff) in all_S_outputs:
        S_matrixes[link_method][cutoff] = S_group

    if scache_only:
        sys.stdout.write(f'Terminating early because scache_only is true, time is {time.time() - time0} \n')
        return None

    # Construct oracle (curse of dimensionality applies here)
    feature_methods = [fname for fname in feature_fns]
    for fname in feature_methods:
        if fname not in feature_fn_kwargs:
            feature_fn_kwargs[fname] = {}
    oracle_results = pd.DataFrame(columns = ORACLE_COLUMNS)

    # Helper function which will be used for multiprocessing ----------------------
    partial_eval_oracles = partial(eval_oracles, 
        n = n, p = p, q = q, X = X, y = y, corr_matrix = corr_matrix, 
        Q = Q, beta = beta, sample_kwargs = sample_kwargs,
        link_methods = link_methods, feature_fns = feature_fns, 
        feature_fn_kwargs = feature_fn_kwargs,
        all_cutoffs = all_cutoffs, all_groups = all_groups, 
        S_matrixes = S_matrixes, time0 = time0, copies = copies,
        compute_split_oracles = compute_split_oracles
    )

    # End helper function ---------------------------
    sys.stdout.write("Picking the best oracles!\n")

    # Don't use the pool object if n-processes is 1
    if num_processes == 1:
        all_outputs_to_add = []
        for j in range(num_data_samples):
            all_outputs_to_add.append(partial_eval_oracles(j))
    else:
        with Pool(num_processes) as thepool:
            all_outputs_to_add = thepool.map(
                partial_eval_oracles, list(range(num_data_samples))
            )

    # Put it all together
    for process_output in all_outputs_to_add:
        for to_add in process_output:
            oracle_results = oracle_results.append(to_add)

    # Pick best cutoffs based on mean power for each oracle
    all_oracle_cutoffs = {}
    for oracle_type in oracle_results['oracle_type'].unique():

        # Create subset, calculate means
        subset_results = oracle_results.loc[oracle_results['oracle_type'] == oracle_type]
        mean_powers = subset_results.groupby(
            ['feature_fn', 'link_method', 'cutoff']
        )['power'].mean()

        # Take max and save
        oracle_cutoffs = mean_powers.unstack().idxmax(1).unstack()
        all_oracle_cutoffs[oracle_type] = oracle_cutoffs

    sys.stdout.write(f'Finished creating oracles: comparing methods, time is {time.time() - time0}\n')

    # Initialize output to actually compare methods
    output_df = pd.DataFrame(columns = FINAL_COLUMNS)

    # Create helper function for multiprocessing
    partial_one_sample_comparison = partial(one_sample_comparison, 
        n = n, p = p, q = q, X = X, y = y, corr_matrix = corr_matrix, 
        Q = Q, beta = beta, sample_kwargs = sample_kwargs,
        links = links, all_oracle_cutoffs = all_oracle_cutoffs,
        link_methods = link_methods, feature_fns = feature_fns, 
        feature_fn_kwargs = feature_fn_kwargs,
        all_cutoffs = all_cutoffs, all_groups = all_groups, 
        S_matrixes = S_matrixes, time0 = time0, copies = copies,
        reduction = reduction
    )

    # Don't use pool object if num_processes == 1
    if num_processes == 1:
        comparisons_to_add = []
        for j in range(num_processes):
            comparisons_to_add.append(partial_one_sample_comparison(j))
    else:
        with Pool(num_processes) as thepool:
            comparisons_to_add = thepool.map(
                partial_one_sample_comparison, list(range(num_data_samples))
            )

    sys.stdout.write(f'Finished: now just combining outputs, time is {time.time() - time0}\n')

    # Combine outputs
    for list_to_add in comparisons_to_add:
        for to_add in list_to_add:
            output_df = output_df.append(to_add, ignore_index = True)
    

    return output_df, oracle_results, S_matrixes



        

