#!/bin/bash
#SBATCH -p shared,janson_cascade #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 47 #Number of cores
#SBATCH -t 0-72:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../../output/nongaussianv3/%j_ar1t_final.out #File to which standard out will be written
#SBATCH -e ../../output/nongaussianv3/%j_ar1t_final.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

# Install python/anaconda, activate base environment
module purge
module load Anaconda3/5.0.1-fasrc02
source activate knockadapt1

# AR1 t 
python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['ar1'] --a_dgp [3] --feature_stat_filter [lasso,dlasso,ridge] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 128 --coeff_size_dgp 0.3 --coeff_dist_dgp uniform --corr_signals_dgp [True,False] --n_sample [350,500,750,1000] --x_dist_sample ar1t --df_t_dgp 3 --df_t_sample 3 --resample_sigma True --resample_beta True --knockoff_type_filter [artk,gaussian] --description Lasso,Ridge on AR1 with p500, linear cond_mean, t_dist design 

# Block t
#python3 ../../degentestv2.py --p_dgp 500 --method_dgp daibarber2016 --gamma_dgp 0 --rho_dgp [0.5] --feature_stat_filter [lasso,dlasso,ridge] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 128 --coeff_size_dgp 0.4 --coeff_dist_dgp uniform --n_sample [350,500,750,1000] --x_dist_sample blockt --df_t_dgp 3 --df_t_sample 3 --resample_sigma True --resample_beta True --knockoff_type_filter [blockt,gaussian] --description Lasso,Ridge on blockt with p500, linear cond_mean

# Ising with metro
#python3 ../../degentestv2.py --p_dgp 625 --x_dist_dgp gibbs --x_dist_sample gibbs --method_dgp ising --method_sample ising --feature_stat_filter [lasso,ridge,dlasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 128 --coeff_size_dgp 0.4 --coeff_dist_dgp uniform --n_sample [350,500,750,1000] --knockoff_type_filter [gaussian,ising] --resample_beta True --resample_sigma False --description Lasso on ising with p625, secnd order with graph lasso, linear cond_mean


module purge

