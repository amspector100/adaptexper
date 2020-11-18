#!/bin/bash
#SBATCH -p janson_cascade #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 47 #Number of cores
#SBATCH -t 0-72:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../../output/nonlinearv3/%j_equi_final.out #File to which standard out will be written
#SBATCH -e ../../output/nonlinearv3/%j_equi_final.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

# Install python/anaconda, activate base environment
module purge
module load Anaconda3/5.0.1-fasrc02
source activate knockadapt1

# --- Nonlinear stuff 

python3 ../../degentestv2.py --p_dgp 200 --method_dgp ['daibarber2016'] --rho_dgp [0.5] --gamma_dgp [0,1] --feature_stat_filter [lasso,randomforest,deeppink] --sparsity_dgp 0.15 --seed_start 0 --cv_score_fstat 0 --num_processes 47 --reps 128 --coeff_size_dgp 5 --cond_mean_sample [cos,quadratic,pairint,cubic,trunclinear] --n_sample [200,300,500,750,1000,2000,3000] --description Lasso + nonlinear fstats on equi p200 large n, quadratic+cos cond means

#python3 ../../degentestv2.py --p_dgp 200 --method_dgp ['ar1'] --a_dgp [3] --feature_stat_filter [randomforest,deeppink,lasso] --sparsity_dgp 0.15 --seed_start 0 --cv_score_fstat 0 --num_processes 47 --reps 128 --coeff_size_dgp 5 --cond_mean_sample [pairint,cubic,trunclinear,cos,quadratic] --corr_signals_dgp [True,False] --resample_beta True --resample_sigma True --n_sample [200,300,500,750,1000,2000,3000] --description Randomforest on AR1 p200 large n, all nonlinear cond means

#python3 ../../degentestv2.py --p_dgp 200 --method_dgp ['ver','qer'] --delta_dgp [0.2] --feature_stat_filter [lasso,randomforest,deeppink]  --sparsity_dgp 0.15 --seed_start 0 --cv_score_fstat 0 --num_processes 47  --reps 128 --coeff_size_dgp 5 --cond_mean_sample [pairint,cubic,trunclinear,cos,quadratic] --resample_beta True --resample_sigma True --n_sample [200,300,500,750,1000,2000,3000] --description Lasso + nonlinear fstats on ver/qer with p200 large n, all nonlinear cond means

module purge

