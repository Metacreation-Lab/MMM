#!/bin/bash

# Inspired from https://github.com/bigscience-workshop/bigscience/blob/7ccf7e42577fe71e88cf8bed3b9ca965c7afb8f7/train/tr11-176B-ml/tr11-176B-ml.slurm

# Set SLURM / hardware environment
#SBATCH --job-name=train-mistral_2
#SBATCH --output=logs/train-mistral_2.out
#SBATCH --error=logs/train-mistral_err_2.out
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per node
#SBATCH --gpus-per-node=v100l:4
#SBATCH --cpus-per-task=10   # nb of CPU cores per task
#SBATCH --mem=100G
#SBATCH --time=12:00:00

# Define args
MODEL_TRAIN_ARGS=" \
    --deepspeed slurm/ds_config.json \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 32 \
    --gradient-accumulation-steps 2 \
    --model MMM_11M_mistral_2 \
    "

# Output GPUs and ram info
echo "START TIME: $(date)"
nvidia-smi
nvidia-smi topo -m
free -h

# Hardware vars
MASTER_HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_HOSTNAME" hostname --ip-address)
MASTER_PORT=9902
echo "Master hostname: $MASTER_HOSTNAME"
echo "Master addr: $MASTER_IP"
echo "Node list: $SLURM_JOB_NODELIST"

# Defining the right environment variables
export PYTHONPATH=$PYTHONPATH:$SCRATCH/MMM
export HF_HOME=$SLURM_TMPDIR/.hf_cache
export HF_METRICS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
# The below variable is required to avoid a warning with the hf tokenizers lib and multiprocessing
# Weirdly, the tokenizer lib is used somewhere before that the dataloader create several workers,
# even when average_num_tokens_per_note is hardcoded in the Dataset class
# https://github.com/huggingface/transformers/issues/5486
# best explanation: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
export TOKENIZERS_PARALLELISM=0
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export TORCH_COMPILE=0

# Move hugging face dataset from scratch to local file system
# This is done on every nodes.
# Docs: https://docs.alliancecan.ca/wiki/Using_node-local_storage
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir $SLURM_TMPDIR/data && cp -r $SCRATCH/data/GigaMIDI $SLURM_TMPDIR/data/"

# Set launcher command with params
#export LAUNCHER="torchrun \
#    --nproc_per_node $SLURM_GPUS_PER_NODE \
#    --nnodes $SLURM_NNODES \
#    --node_rank $SLURM_PROCID \
#    --rdzv_endpoint $MASTER_IP:$MASTER_PORT \
#    --rdzv_backend c10d \
#    --max_restarts 0 \
#    --role $SLURMD_NODENAME: \
#    --tee 3 \
#    "
# Replace with line below when using one unique node
export LAUNCHER="torchrun --nproc_per_node $SLURM_GPUS_PER_NODE"

# Load the python environment
module load gcc arrow/17.0.0 cudacore/.12.6.2 cudacompat/.12.6 # needed since arrow can't be installed in the venv via pip
source .venv/bin/activate

# Run the training
# Tensorboard can be access by running (with computenode replaced with the node hostname):
# ssh -N -f -L localhost:6006:computenode:6006 userid@cedar.computecanada.ca
tensorboard --logdir=runs --host 0.0.0.0 --load_fast false & srun --jobid "$SLURM_JOBID" bash -c "$LAUNCHER scripts/train_model.py $MODEL_TRAIN_ARGS"

echo "END TIME: $(date)"
