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
#SBATCH --time=3:00:00

echo "START TIME: $(date)"

VENV=".venv/"
module purge

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment"
  module load python/3.11
  virtualenv .venv
fi

# Load the python environment
module load gcc arrow/17.0.0 rust  # needed since arrow can't be installed in the venv via pip
source .venv/bin/activate

pip install symusic==0.5.0 -vv
pip install git+https://github.com/DaoTwenty/MidiTok@expressive -vv
pip install transformers==4.49.0 accelerate==1.4.0 tensorboard==2.19.0 -vv
pip install flash_attn==2.5.7 -vv
pip install deepspeed==0.14.4 -vv
pip install datasets==3.3.2 -vv
pip install triton==3.1.0 -vv
pip install .
pip freeze

echo "END TIME: $(date)"
