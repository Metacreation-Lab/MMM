#!/bin/bash

# Set SLURM / hardware environment
#SBATCH --job-name=install
#SBATCH --output=logs/install.out
#SBATCH --error=logs/install_err.out
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per node
#SBATCH --cpus-per-task=10   # nb of CPU cores per task
#SBATCH --mem=30G
#SBATCH --time=2:00:00

# Load the python environment
module load gcc arrow/17.0.0 rust  # needed since arrow can't be installed in the venv via pip
source .venv/bin/activate

srun bash -c "pip install symusic==0.5.0"
srun bash -c "pip install git+https://github.com/DaoTwenty/MidiTok@expressive"
srun bash -c "pip install transformers accelerate tensorboard"
srun bash -c "pip install flash_attn"
srun bash -c "pip install deepspeed==0.14.4"
srun bash -c "pip install datasets"
srun bash -c "pip list"


echo "END TIME: $(date)"
