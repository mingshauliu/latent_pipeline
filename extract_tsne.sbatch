#!/bin/bash
# Extract FM-encoder latents from the NEW best dim8 FM (ffdsq458 ep486, val 0.012231),
# fit N(mu,Sigma) + corner plot, then t-SNE coloured by suite. New _ep486 outputs so
# the old sv6mp9wt latent_stats/tsne are preserved for comparison.
#   sbatch extract_tsne.sbatch
#
#SBATCH -J extract_tsne
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=latent-pipeline/ffdsq458/checkpoints/best-epoch=486-val_loss=0.012231.ckpt
NPZ=cached/latent_stats_ep486.npz
echo "Node: $(hostname) | ckpt: $CKPT"
nvidia-smi

# config/config.yaml -> latent_dim=8, variational=False (matches ffdsq458 arch).
python extract_latents.py \
    --config config/config.yaml \
    --checkpoint "$CKPT" \
    --out "$NPZ"

# t-SNE (CPU-only; runs fine on the GPU node).
python tsne_latent.py \
    --npz "$NPZ" \
    --out cached/latent_tsne_ep486.png
