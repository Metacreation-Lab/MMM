#!/bin/bash

# Set SLURM / hardware environment
#SBATCH --job-name=midi-audio-matching
#SBATCH --output=logs/midi-audio-matching.out
#SBATCH --error=logs/midi-audio-matching_err.out
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per node
#SBATCH --cpus-per-task=16    # nb of CPU cores per task
#SBATCH --mem=1600G
#SBATCH --time=24:00:00

# Output ram info
echo "START TIME: $(date)"
free -h

# Defining the right environment variables
export PYTHONPATH=$HOME/MMM
export HF_HOME=$SCRATCH/.hf_cache

# Load the python environment
# Make sure the required packages are installed
source .venv/bin/activate

# Run the training
python utils/GigaMIDI/metamidi_md5_audio_matching.py

echo "END TIME: $(date)"
