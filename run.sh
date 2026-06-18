#!/bin/bash
# Usage:
#   sbatch run.sh train    [config.yaml]
#   sbatch run.sh infer    [config.yaml]
#   sbatch run.sh nf_train [config.yaml] [overrides...]
#   sbatch run.sh nf_infer [config.yaml] [overrides...]

#SBATCH -J LATENT
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH -t 0-02:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs

MODE="${1:-train}"
CONFIG="${2:-config/config.yaml}"

echo "Node: $(hostname) | Mode: $MODE | Config: $CONFIG"
nvidia-smi

case "$MODE" in
  train)
    srun python train.py --config "$CONFIG"
    ;;
  infer)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python infer.py --config "$CONFIG"
    ;;
  nf_train)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    srun python -m nf.train --config "$CONFIG" "${@:3}"
    ;;
  nf_infer)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -m nf.infer --config "$CONFIG" "${@:3}"
    ;;
  nf_ood)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -m nf.ood --config "$CONFIG" "${@:3}"
    ;;
  *)
    echo "Unknown mode: $MODE (train | infer | nf_train | nf_infer | nf_ood)"
    exit 1
    ;;
esac
