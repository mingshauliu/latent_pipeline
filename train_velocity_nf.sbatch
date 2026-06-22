#!/bin/bash
# Standalone velocity-NF: p(Omega_m, sigma_8 | v) where v = Mgas_norm - Nbody_norm
# (the FM 'velocity' field, clean). Trained on the FM cache pool (cfg['data']); vel
# stats cached to cached/norm_velocity.npz (shared with the FM aux critic). This NF is
# then FROZEN and used ONLY as the FM cosmo-consistency aux critic (NOT a comparison
# eval harness). Same split/seed (42) + eff-batch (8x2x4=64) as nf_ck_real.
#   sbatch train_velocity_nf.sbatch
#
#SBATCH -J vel_nf
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH -t 1-00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs cached nf_ck_velocity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Node: $(hostname) | velocity-NF train"
nvidia-smi

srun python -m nf.train \
    --config config/config.yaml \
    --data_mode velocity \
    --checkpoint_dir nf_ck_velocity \
    --run_name velocity_nf
