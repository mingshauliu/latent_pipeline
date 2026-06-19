#!/bin/bash
# Latent corner plot for the det_dim3 sweep winner (latent_dim=3, deterministic).
# Encodes the 3000-cube training Mgas cache with the det3 EMA gas_encoder, fits a
# Gaussian, writes cached/latent_stats_det3.npz + cached/latent_stats_det3_corner.png.
#
#   sbatch experiments/extract_det3.sbatch
#
#SBATCH -J det3_latents
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH -t 0-00:30
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs cached

# det3 sweep winner: lowest-val ckpt (ep008 0.014682); ep021/ep026 drifted up.
CKPT=latent-pipeline-latentsweep/vmve93wg/checkpoints/best-epoch=008-val_loss=0.014682.ckpt

echo "Node: $(hostname) | ckpt: $CKPT"
nvidia-smi

# --config = det_dim3.yaml so the model builds with latent_dim=3, variational=false
# (config/config.yaml is latent_dim=8 -> would shape-mismatch on load).
python extract_latents.py \
    --config experiments/latent_sweep/det_dim3.yaml \
    --checkpoint "$CKPT" \
    --out cached/latent_stats_det3.npz
