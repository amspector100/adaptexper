#!/bin/bash
#SBATCH -p janson_cascade #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 47 #Number of cores
#SBATCH -t 0-72:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../../output/linearv3/%j_dsliu2020.out #File to which standard out will be written
#SBATCH -e ../../output/linearv3/%j_dsliu2020.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

# Install python/anaconda, activate base environment
module purge
module load Anaconda3/5.0.1-fasrc02
source activate knockadapt1

# -- Linear stuff, in particular AR1

python3 ../../degentestv2.py --p_dgp 500 --n_dgp 500 --n_sample 500  --method_dgp ['daibarber2016'] --rho_dgp [0,0.8] --gamma_dgp 1 --feature_stat_filter lasso --use_lars_fstat True --y_dist_sample gaussian --sparsity_sample 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 128 --coeff_dist_sample dsliu2020 --resample_beta True --q_filter 0.1 --description DSLIU2020 REPLICATION p500

python3 ../../degentestv2.py --p_dgp 2000 --n_dgp 500 --n_sample 500  --method_dgp ['daibarber2016'] --rho_dgp [0,0.8] --gamma_dgp 1 --feature_stat_filter lasso --use_lars_fstat True --y_dist_sample gaussian --sparsity_sample 0.025 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 47 --coeff_dist_sample dsliu2020 --resample_beta True --q_filter 0.1 --description DSLIU2020 REPLICATION p2000


module purge

