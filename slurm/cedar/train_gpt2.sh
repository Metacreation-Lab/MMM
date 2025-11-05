#!/bin/bash

#SBATCH --job-name=train-gpt2
#SBATCH --output=logs/train-gpt2.out
#SBATCH --error=logs/train-gpt2_err.out
#SBATCH --account=def-pasquier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # One task per MIG slice
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3:0:0

# Model args
MODEL_TRAIN_ARGS=" \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 64 \
    --gradient-accumulation-steps 2 \
    --model MMM_gpt2 \
"

# Hardware info
echo "START TIME: $(date)"
nvidia-smi
free -h

# Python / env setup
module purge
module load gcc arrow rust
source .venv/bin/activate
module list

export PYTHONPATH=$PYTHONPATH:$SCRATCH/MMM
export HF_HOME=$SLURM_TMPDIR/.hf_cache
export HF_METRICS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=0

# Move dataset to local node storage
srun --ntasks=1 bash -c "mkdir -p $SLURM_TMPDIR/data/GigaMIDI && cp -r $SCRATCH/data/GigaMIDI/v* $SLURM_TMPDIR/data/GigaMIDI"

# Launch training: one process per MIG
srun --ntasks=4 --ntasks-per-node=1 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  echo "Task $SLURM_PROCID using MIG $CUDA_VISIBLE_DEVICES"
  tensorboard --logdir=runs --host 0.0.0.0 --load_fast false &
  python scripts/train_model.py $MODEL_TRAIN_ARGS
'

echo "END TIME: $(date)"
