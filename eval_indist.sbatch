#!/bin/bash
# In-distribution sanity for the det3 FM (latent_dim=3): imshow + truth-vs-pred.
#   sbatch eval_indist.sbatch
#
#SBATCH -J det3_indist
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
mkdir -p logs outputs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=latent-pipeline-latentsweep/vmve93wg/checkpoints/best-epoch=008-val_loss=0.014682.ckpt
echo "Node: $(hostname) | ckpt: $CKPT"
nvidia-smi

# encode = oracle in-distribution latent (conditions on the true-Mgas latent).
python eval_indist.py \
    --config experiments/latent_sweep/det_dim3.yaml \
    --checkpoint "$CKPT" \
    --suite IllustrisTNG --n 4 --latent_mode encode
