#!/bin/bash
# GPU smoke for the ne multi-task output channel: baseline backward-compat + ne flow
# (out_channels=2) + warm-load 1->2ch from the real ep486 ckpt. Fast (<2 min).
#   sbatch gpu_smoke_ne.sbatch
#
#SBATCH -J smoke_ne
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-00:20
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs
echo "Node: $(hostname)"
nvidia-smi
srun python -u smoke_ne.py
