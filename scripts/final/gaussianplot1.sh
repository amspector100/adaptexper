#!/bin/bash
#SBATCH -p janson_cascade #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 47 #Number of cores
#SBATCH -t 0-72:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../../output/linearv3/%j_ci_er.out #File to which standard out will be written
#SBATCH -e ../../output/linearv3/%j_ci_er.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

# Install python/anaconda, activate base environment
module purge
module load Anaconda3/5.0.1-fasrc02
source activate knockadapt1

# -- Linear stuff, in particular AR1

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['dirichlet'] --temp_dgp [1] --feature_stat_filter [lasso,dlasso,ridge] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 32 --reps 256 --coeff_size_dgp 1 --description Lasso,Ridge,Random Forest, Deeppink on dirichlet with p500 and many cond_means

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['ver','qer'] --delta_dgp [0.2] --feature_stat_filter [lasso,dlasso,ridge] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 32 --reps 128 --coeff_size_dgp 1 --coeff_dist_dgp uniform --description GAUSSIAN PLOT1: OLS,Lasso,Ridge,Dlasso on TRUE ER with p500 and linear cond_means. Maxent knockoffs too

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['ar1'] --a_dgp 3 --feature_stat_filter [lasso,ridge,dlasso] --coeff_size_dgp 2 --reps 128 --num_processes 47 --cv_score_fstat 0 --seed_start 0 --coeff_dist_dgp uniform --corr_signals_dgp [True,False] --sparsity_dgp 0.1 --description GAUSSIAN PLOT1: Linear feature statistics on AR1. Maxent knockoffs too

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['daibarber2016'] --rho_dgp [0.5] --gamma_dgp [0,1] --feature_stat_filter [lasso,ridge,dlasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 128 --coeff_size_dgp 1 --coeff_dist_dgp uniform --description GAUSSIAN PLOt1: dlasso,Lasso,Ridge on Equicorr/Block equicorr with p500. Maxent knockoffs too


#python3 ../../degentestv2.py --p_dgp 500 --method_dgp 'dirichlet' --feature_stat_filter [ols,lasso,ridge,dlasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 32 --reps 47 --coeff_size_dgp 1 --coeff_dist_dgp uniform --description Gaussian plot1: linear statistics on dirichlet with p500, maxent knockoffs too

module purge

