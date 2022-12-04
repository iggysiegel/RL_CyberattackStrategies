#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name=train
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o train.out
#SBATCH -e train.err

# Loading the required module
source /etc/profile
module load anaconda/2022a

# Run the script
python train.py