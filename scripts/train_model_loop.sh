#!/bin/bash

# Script to train a model while sending consecutive Slurm jobs

# Parse arguments
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

# Set vars
TEST_RESULTS_FILE="runs/${MODEL}/train_results.json"
COUNT=0

# Loop job until training is done
while [ ! -f "$TEST_RESULTS_FILE" ];
do
    sbatch --wait slurm/train_model.sh --model ${MODEL}
    mv "logs/train-mmm.out" "logs/train-mmm_$COUNT.out"
    mv "logs/train-mmm_err.out" "logs/train-mmm_${COUNT}_err.out"
    ((COUNT++))
done
