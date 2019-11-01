#!/bin/bash
#SBATCH -p janson #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 1 #Number of cores
#SBATCH -t 0-00:05 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=500 #Memory per cpu in MB (see also --mem)
#SBATCH -o output/slurm.%N.%j.out #File to which standard out will be written
#SBATCH -e output/slurm.%N.%j.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

# Install python/anaconda, activate base environment
module purge
module load Anaconda3/5.0.1-fasrc02
source activate base

# Install group_lasso and run file
pip install group_lasso
python3 main.py --n 50 --p 20 --num_datasets 10 --covmethod 'ErdosRenyi' 

module purge

# Just use SCP on terminal
