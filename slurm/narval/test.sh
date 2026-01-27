#!/bin/bash

# Load the python environment
module load gcc arrow/17.0.0 # needed since arrow can't be installed in the venv via pip
source .venv/bin/activate

python scripts/testing.py --model MMM_gpt2 --workers 8