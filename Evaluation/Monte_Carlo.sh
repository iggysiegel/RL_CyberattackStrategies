#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name=monte_carlo
#SBATCH -n 20
#SBATCH -a 1-10
#SBATCH -o monte_carlo_%a.out
#SBATCH -e monte_carlo_%a.err

# Loading the required module
source /etc/profile
module load anaconda/2022a

# Run the script
python Monte_Carlo.py