#!/bin/bash
# SEPARATE FM model: N-body VELOCITY as an extra input channel -> push xcorr up.
# encoder3D reaches xcorr ~0.9 partly via N-body velocity conditioning (its UNet
# in_channels=5); ours was density-only (in_channels=2) -> under-shoots high-k P(k)
# (oracle xcorr 0.686, P(k) ratio 0.84 @k<15). This run adds the voxelised DM
# velocity field as a 3rd static conditioning channel ([x_t, nbody, vel]).
#
# Warm-starts (PARTIAL, init_strict=false) from the dim8 best FM ep486 (val 0.012231):
# the full UNet backbone + encoder copy verbatim; net.enc1.conv1 / enc1.skip expand
# their input dim 2->3 (the new vel col starts fresh, others copied); cond_fuse copies
# time+cosmo cols. Fresh optimiser, cosine T_max=150 lr 2e-4->1e-6 over 150 ep,
# noise_std 0.1 (matches ep486 -> smooth warm-start). wandb latent-pipeline-velocity.
#
# PREREQS (run first):
#   1) voxelise/voxelise_nbody_vel.sh {IllustrisTNG,Astrid,SIMBA}  -> per-cube grids
#      then `python voxelise_nbody_vel.py --suite all --stack`     -> Grids_Vcdm_Nbody
#   2) python prep_cache.py --with_vel   -> NbodyVel_norm_{suite} in the FM cache
#                                           + nbodyvel stats in norm_meta/norm_latent
#
#   sbatch experiments/velocity/fm_velocity.sbatch
#
#SBATCH -J fm_velocity
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH -t 1-12:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs

echo "Node: $(hostname) | FM + N-body velocity channel"
nvidia-smi

srun python train.py --config experiments/velocity/fm_velocity.yaml
