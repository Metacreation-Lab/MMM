#!/bin/bash

# Set SLURM / hardware environment
#SBATCH --job-name=train-tokenizer
#SBATCH --output=logs/train-tokenizer.out
#SBATCH --error=logs/train-tokenizer_err.out
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per node
#SBATCH --cpus-per-task=16    # nb of CPU cores per task
#SBATCH --mem=64G
#SBATCH --time=20:00:00

# Output ram info
echo "START TIME: $(date)"
free -h

# Defining the right environment variables
export PYTHONPATH=$PYTHONPATH:$SCRATCH/MMM
export HF_HOME=$SCRATCH/.hf_cache

# Load the python environment
# Make sure the required packages are installed
source .venv/bin/activate

# Run the training
python scripts/train_tokenizer.py

echo "END TIME: $(date)"
