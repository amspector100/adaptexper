#!/bin/bash
#SBATCH -p janson_cascade #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 47 #Number of cores
#SBATCH -t 0-72:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../../output/linearv3/%j_equiplot1_part2.out #File to which standard out will be written
#SBATCH -e ../../output/linearv3/%j_equiplot1_part2.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

# Install python/anaconda, activate base environment
module purge
module load Anaconda3/5.0.1-fasrc02
source activate knockadapt1

#python3 ../../degentestv2.py --p_dgp 500 --n_sample [250,500,750,1000]  --method_dgp ['daibarber2016'] --rho_dgp start0end0.4numvals5 --gamma_dgp 1 --feature_stat_filter [lasso,ridge] --y_dist_sample [gaussian,binomial] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 256 --coeff_size_dgp 1 --coeff_dist_dgp uniform --description PLOT1 FOR EQUICORR FOR RHOleq0.4, all linear stats

python3 ../../degentestv2.py --p_dgp 500 --n_sample [250,500,750,1000]  --method_dgp ['daibarber2016'] --rho_dgp start0.5end0.9numvals5 --gamma_dgp 1 --feature_stat_filter [lasso,ridge] --y_dist_sample [gaussian,binomial] --sparsity_dgp 0.1 --cv_score_fstat 0 --seed_start 0 --num_processes 47 --reps 128 --coeff_size_dgp 1 --coeff_dist_dgp uniform --description PLOT1 FOR EQUICORR FOR RHOgeq0.5, all linear stats



module purge

