#!/bin/bash
# Continue the dim8 FM (sv6mp9wt) — it was still improving at ep276 (best ep269
# val 0.013353, not plateaued). resume_from=sv6mp9wt/last.ckpt restores
# optimiser+scheduler+epoch (277) -> continues the cosine toward ep2000.
# 4x A100 DDP to keep the original effective batch (batch2 x accum4 x devices4 = 32).
#   sbatch resume_dim8.sbatch
#
#SBATCH -J dim8_resume
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH -t 2-00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs

echo "Node: $(hostname) | resume sv6mp9wt last.ckpt (ep276) -> cosine toward ep2000"
nvidia-smi

# config/config.yaml: devices=4, resume_from=sv6mp9wt/last.ckpt (set 2026-06-19).
srun python train.py --config config/config.yaml
