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
#SBATCH --time=3-00:00:00

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p $SLURM_TMPDIR/data && cp -r -v $SCRATCH/data/GigaMIDI $SLURM_TMPDIR/data/"

# Output ram info
echo "START TIME: $(date)"
free -h

module load python/3.11
module load gcc arrow/17.0.0 rust

source .venv/bin/activate

# Defining the right environment variables
export PYTHONPATH=$PYTHONPATH:$SCRATCH/MMM
export HF_HOME=$SCRATCH/.hf_cache

# Load the python environment
# Make sure the required packages are installed

# Run the training
python scripts/train_tokenizer.py --model $MODEL

echo "END TIME: $(date)"
