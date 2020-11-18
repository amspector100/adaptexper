#!/bin/bash
#SBATCH -p shared,janson_cascade #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 47 #Number of cores
#SBATCH -t 0-72:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../../output/fxv3/%j_er_final.out #File to which standard out will be written
#SBATCH -e ../../output/fxv3/%j_er_final.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

# Install python/anaconda, activate base environment
module purge
module load Anaconda3/5.0.1-fasrc02
source activate knockadapt1

# ---- FX knockoffs

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['daibarber2016'] --rho_dgp [0.5] --gamma_dgp [0,1] --feature_stat_filter [lasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --zstat_fstat lars_path --num_processes 47 --reps 128 --coeff_size_dgp 0.15 --coeff_dist_dgp uniform --n_sample [1005,1250,1500,1750,2000] --fixedx_filter True --description Lasso on equicorr\block equicorr with p500, linear cond_mean, FX knockoffs and low coeff_size

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['ar1'] --a_dgp [3] --feature_stat_filter [lasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --zstat_fstat lars_path --num_processes 47 --reps 128 --coeff_size_dgp 0.25 --coeff_dist_dgp uniform --corr_signals_dgp [True,False] --n_sample [1005,1250,1500,1750,2000] --fixedx_filter True --description Lasso on AR1 with p500, linear cond_mean, FX knockoffs, low coeff size

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['dirichlet'] --temp_dgp 1 --feature_stat_filter [lasso,ridge] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --zstat_fstat lars_path --num_processes 32 --reps 128 --coeff_size_dgp 0.75 --coeff_dist_dgp uniform --n_sample [1000,1200,1500,2000] --fixedx_filter True --description Lasso, ridge, dlasso dirichlet with p500, FX knockoffs, unif coeff dist

python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['ver'] --delta_dgp [0.2] --feature_stat_filter [lasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --zstat_fstat lars_path --num_processes 47 --reps 128 --coeff_size_dgp 0.5 --coeff_dist_dgp uniform --n_sample [1005,1250,1500,1750,2000] --fixedx_filter True --description Linear stats on VER with p500, linear cond_mean, FX knockoffs

python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['qer'] --delta_dgp [0.2] --feature_stat_filter [lasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --zstat_fstat lars_path --num_processes 47 --reps 128 --coeff_size_dgp 0.1 --coeff_dist_dgp uniform --n_sample [1005,1250,1500,1750,2000] --fixedx_filter True --description Linear stats on QER with p500, linear cond_mean, FX knockoffs


module purge

