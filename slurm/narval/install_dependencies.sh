#!/bin/bash

echo "START TIME: $(date)"

VENV=".venv/"
module purge

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment"
  module load python/3.11
  virtualenv .venv
fi

# Load the python environment
module purge
module load gcc arrow rust # needed since arrow can't be installed in the venv via pip
source .venv/bin/activate

pip install "huggingface-hub[cli]==0.34.0" \
            miditok \
            transformers==4.48.0 accelerate==1.10.0 tensorboard==2.15.0 \
            deepspeed \
            datasets==3.6.0 \
            triton==3.2.0

pip list

echo "END TIME: $(date)"
