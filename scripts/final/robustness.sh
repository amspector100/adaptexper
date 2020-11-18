#!/bin/bash
#SBATCH -p shared,janson #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 32 #Number of cores
#SBATCH -t 0-72:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../../output/infv3/%j_er_final.out #File to which standard out will be written
#SBATCH -e ../../output/infv3/%j_er_final.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

# Install python/anaconda, activate base environment
module purge
module load Anaconda3/5.0.1-fasrc02
source activate knockadapt1

# ---- Inferring Sigma

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['daibarber2016'] --rho_dgp [0.6] --gamma_dgp 0 --feature_stat_filter [lasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 32 --reps 128 --coeff_size_dgp 0.5 --coeff_dist_dgp uniform  --n_sample [200,350,500,650,875,1000] --infer_sigma_filter True --shrinkage_filter [none,graphicallasso,ledoitwolf] --description Lasso on Block equicorr with p500, linear cond_mean, but infer Cov matrix, low coeff size

# NO graphical lasso for equicorrelated
#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['daibarber2016'] --rho_dgp [0.5,0.6] --gamma_dgp 1 --feature_stat_filter [lasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 32 --reps 128 --coeff_size_dgp 0.5 --coeff_dist_dgp uniform  --n_sample [200,350,500,650,875,1000] --infer_sigma_filter True --shrinkage_filter [none,ledoitwolf] --description Lasso on Equicorr equicorr with p500, linear cond_mean, but infer Cov matrix, low coeff size

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['ar1'] --a_dgp [3] --feature_stat_filter [lasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 32 --reps 128 --coeff_size_dgp 1 --coeff_dist_dgp uniform --n_sample [200,350,500,650,875,1000] --corr_signals_dgp [True,False] --infer_sigma_filter True --shrinkage_filter [none,graphicallasso,ledoitwolf] --description Lasso,Ridge on AR1 with p500, linear cond_mean, infer cov

python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['ver','qer'] --delta_dgp [0.2] --feature_stat_filter [lasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 32 --reps 128 --coeff_size_dgp 0.5 --coeff_dist_dgp uniform --n_sample [200,350,500,650,875,1000] --infer_sigma_filter True --shrinkage_filter [none,graphicallasso,ledoitwolf] --coeff_dist_dgp uniform --description Lasso,Ridge on ER with p500, linear cond_mean, infer cov

#python3 ../../degentestv2.py --p_dgp 500 --method_dgp ['dirichlet'] --temp_dgp 1 --feature_stat_filter [lasso,ridge,dlasso] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 32 --reps 128 --coeff_size_dgp 3 --n_sample [350,500,750,1000] --infer_sigma_filter True --shrinkage_filter [none,graphicallasso,ledoitwolf] --description Lasso,Ridge on dirichlet with p500, infer cov

module purge

