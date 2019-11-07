""" Old Oracle was too powerful """
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
            sys.stdout.write(f'Beginning trial {j}, method {method_name}, time is {time.time() - time0}')
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