#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name=optuna_ppo
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o optuna_ppo.out
#SBATCH -e optuna_ppo.err

# Loading the required module
source /etc/profile
module load anaconda/2022a

# Run the script
python optuna_ppo.py