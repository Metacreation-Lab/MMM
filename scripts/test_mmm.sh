#!/bin/bash

# Set SLURM / hardware environment
#SBATCH --job-name=test_data_loader
#SBATCH --output=logs/test_data_loader.out
#SBATCH --error=logs/test_data_loader_error.out
#SBATCH --account=def-pasquier
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --cpus-per-task=32   # nb of CPU cores per task
#SBATCH --mem=100G
#SBATCH --time=3:00:00

module purge

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir $SLURM_TMPDIR/data && cp -r $SCRATCH/data/GigaMIDI $SLURM_TMPDIR/data/"

export PYTHONPATH=$PYTHONPATH:$SCRATCH/MMM

module load gcc arrow/17.0.0 cuda/12.2
source .venv/bin/activate

ARGS=" \
    --model MMM_epl_mistral \
    --version v3.0.0 \
    --samples 1000 \
    "

python scripts/test_data_loader.py $ARGS
