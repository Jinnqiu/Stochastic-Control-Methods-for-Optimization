#!/bin/bash
#SBATCH --job-name=snake-horse-0301
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ---------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------
# Load modules
module load python/anaconda3

# Activate conda environment
source activate snake_env

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
# Print GPU info
nvidia-smi

# Set memory allocation config to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting simulation for snake-horse-0301.py on A100..."
echo "Running on node: $HOSTNAME"
echo "Date: $(date)"

# Run the python script
python snake-horse-0301.py

echo "Simulation finished at $(date)"
