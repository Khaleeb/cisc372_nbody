#!/bin/bash -l
#SBATCH --job-name=pNbody
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output pNbody-job_%j.out
#SBATCH --error pNbody-job_%j.err
#SBATCH --partition=gpu-v100

# Start my application
srun pNbody
