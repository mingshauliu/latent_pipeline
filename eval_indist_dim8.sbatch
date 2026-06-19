#!/bin/bash
# In-distribution sanity for the OLD dim8 FM (latent_dim=8) — compare vs det3.
# sv6mp9wt best = ep269 val 0.013353 (kept training past the 0.0154 plateau).
#   sbatch eval_indist_dim8.sbatch
#
#SBATCH -J dim8_indist
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
mkdir -p logs outputs/dim8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=latent-pipeline/sv6mp9wt/checkpoints/best-epoch=269-val_loss=0.013353.ckpt
echo "Node: $(hostname) | ckpt: $CKPT"
nvidia-smi

# config/config.yaml -> latent_dim=8, variational=False (the old model arch).
python eval_indist.py \
    --config config/config.yaml \
    --checkpoint "$CKPT" \
    --suite IllustrisTNG --n 4 --latent_mode encode \
    --noise_std 0.1 \
    --out_dir outputs/dim8
