#!/bin/bash
# GPU smoke for the velocity-NF aux loss (Part 2). Validates, at production size:
#   A) velocity-NF training path (nf.train --data_mode velocity --fast_dev_run)
#   B) FM aux-loss wiring + backward-compat (smoke_vel_aux.py, 128^3)
# Neither needs a trained velocity-NF (A is 1-batch; B uses a random critic for plumbing).
#   sbatch gpu_smoke_vel.sbatch
#
#SBATCH -J vel_smoke
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-00:20
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs cached
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Node: $(hostname)"; nvidia-smi

echo "===== A) velocity-NF training path (1-batch fast_dev_run) ====="
python -m nf.train --config config/config.yaml --data_mode velocity \
    --fast_dev_run || exit 1

echo "===== B) FM aux-loss wiring + backward-compat (128^3) ====="
python smoke_vel_aux.py --config config/config.yaml --device cuda
