#!/bin/bash
#SBATCH -p janson_cascade #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 47 #Number of cores
#SBATCH -t 0-72:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../../output/linearv3/%j_ar1plot_rho.out #File to which standard out will be written
#SBATCH -e ../../output/linearv3/%j_ar1plot_rho.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

# Install python/anaconda, activate base environment
module purge
module load Anaconda3/5.0.1-fasrc02
source activate knockadapt1

# -- Linear stuff, in particular AR1

python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['ar1'] --rho_dgp [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] --feature_stat_filter [lasso,ridge,dlasso] --coeff_size_dgp 1 --reps 128 --num_processes 47 --cv_score_fstat 0 --seed_start 0 --coeff_dist_dgp uniform --corr_signals_dgp [True,False] --sparsity_dgp 0.1 --description GAUSSIAN PLOT1: Linear feature statistics on AR1. Maxent knockoffs too

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['ar1'] --a_dgp [0.5,1,2,3] --feature_stat_filter [lasso,ridge,dlasso] --coeff_size_dgp 2 --reps 128 --num_processes 47 --cv_score_fstat 0 --seed_start 0 --coeff_dist_dgp uniform --corr_signals_dgp [True,False] --sparsity_dgp 0.1 --description GAUSSIAN PLOT1: Linear feature statistics on AR1. Maxent knockoffs too

module purge

