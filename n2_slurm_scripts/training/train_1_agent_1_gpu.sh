#!/bin/bash 

# Project
#SBATCH -A hpc-prf-radioml

# Job time limit [days-hours]
#SBATCH -t 0-12

# Resources
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32G
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1

# Select working directory on PFS
#SBATCH --chdir=/scratch/hpc-prf-radioml/felix

# Configure Weights & Biases client
# Remember to define WANDB_BASE_URL and WANDB_API_KEY as well (e.g., in your ~/.bashrc)
WORKING_DIR="$PC2PFS"/hpc-prf-radioml/felix
export WANDB_DIR="$WORKING_DIR"
export WANDB_CONFIG_DIR="$WORKING_DIR"/wandb/.config
export WANDB_CACHE_DIR="$WORKING_DIR"/wandb/.cache

# Configure training
#export DATASET_PATH_RADIOML="$PC2DATA"/hpc-prf-radioml/datasets/RadioML
export DATASET_PATH_RADIOML="$PC2PFS"/hpc-prf-radioml/datasets/RadioML

# Activate Python virtual environment for training
module load lang/Python/3.9.6-GCCcore-11.2.0
source venv_training/bin/activate

# Start 1 agent
SWEEP_ID=$1
CUDA_VISIBLE_DEVICES=0 wandb agent felixj/RadioML/$SWEEP_ID
