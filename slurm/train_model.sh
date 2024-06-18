#!/bin/bash

# Inspired from https://github.com/bigscience-workshop/bigscience/blob/7ccf7e42577fe71e88cf8bed3b9ca965c7afb8f7/train/tr11-176B-ml/tr11-176B-ml.slurm

# Set SLURM / hardware environment
#SBATCH --job-name=train-mmm
#SBATCH --output=logs/train-mmm.out
#SBATCH --error=logs/train-mmm_err.out
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per node
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=48   # nb of CPU cores per task
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00

# Define args
MODEL_TRAIN_ARGS=" \
    --deepspeed \
    --per-device-batch-size 8 \
    --per-device-batch-size-test 16 \
    "

# Output GPUs and ram info
echo "START TIME: $(date)"
nvidia-smi
nvidia-smi topo -m
free -h

# Hardware vars
GPUS_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=9902
echo "Master addr: $MASTER_ADDR"
echo "Node list: $SLURM_JOB_NODELIST"

# Defining the right environment variables
export PYTHONPATH=$HOME/MMM
export HF_HOME=$SCRATCH/.hf_cache
export HF_METRICS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
#export OMP_NUM_THREADS=1

# Set launcher command with params
export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role $SLURMD_NODENAME: \
    --tee 3 \
    "

# Load the python environment
source .venv/bin/activate
# pip install flash-attn --no-build-isolation

# Run the training
srun --jobid "$SLURM_JOBID" bash -c "$LAUNCHER exp_generation.py $MODEL_TRAIN_ARGS"

echo "END TIME: $(date)"
