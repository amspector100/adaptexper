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

# Block-equicorrelated with high rho, low blocksize
#python3 ../../degentestv2.py --p_dgp 500 --n_dgp [375] --coeff_size_dgp [0.5,0.6,0.7,0.8,0.9,1] --method_dgp ['daibarber2016'] --rho_dgp start0.75end0.95numvals5 --gamma_dgp 0 --group_size_dgp 2 --feature_stat_filter [lasso,ridge] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 512 --coeff_dist_dgp none --description CICOMP1: Lasso,Ridge on Block equicorr with p500

# Factor models
#python3 ../../degentestv2.py --p_dgp 500 --method_dgp 'factor' --rank_dgp [1,5,20,50,100] --feature_stat_filter [lasso,ridge] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 128 --coeff_size_dgp 1 --coeff_dist_dgp uniform --description Lasso,Ridge on factor model with p500. CI knockoffs too

# ErdosRenyi
python3 ../../degentestv2.py --p_dgp 500 --method_dgp 'ver' --delta_dgp [0.01,0.05,0.10,0.15,0.2] --feature_stat_filter [lasso,ridge] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 128 --n_dgp 375 --coeff_size_dgp [0.5,0.75,1] --coeff_dist_dgp none --description Lasso,Ridge on VER with p500

module purge

~                                                               
