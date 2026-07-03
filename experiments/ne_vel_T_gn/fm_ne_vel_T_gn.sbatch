#!/bin/bash
# FULL multi-task FM: input [nbody_density, nbody_vel] -> output [Mgas, ne, T]
# (encoder3D parity) with GROUPNORM.
#
# WHY (2026-07-01): the ne_vel_T PIXEL run is a near-exact encoder3D replica (3 targets
# Mgas,ne,T + live FiLM + mlp/no-tanh head + fresh train + base40 encoder + velocity +
# logit-normal) yet STILL collapsed: latent eff-dim 1.49/8 at ep20 == the old ep486 1.48.
# So collapse is NOT objective/head/target/FiLM-init. The LAST un-ported difference is
# the UNet norm: ours = PixelNorm (per-voxel channel-vector renorm, no affine, no spatial
# avg) which cancels the per-channel FiLM modulation the latent carries -> weak
# latent->output gradient -> encoder starves -> collapse. encoder3D uses GroupNorm(1) only
# (7x, zero PixelNorm) -> FiLM's per-channel pattern survives. This run flips ONLY
# norm_type: pixel -> group to isolate that. Everything else == ne_vel_T pixel.
#
#   in_channels = 5 = x_t(3 = Mgas,ne,T) + nbody(1) + vel(1);  out_channels = 3.
#   Encoder encodes the 3 targets (Mgas,ne,T) -> 8-dim, latent_head=mlp (no tanh), base40.
#   Fresh train (init_from null, init_strict true), live FiLM/out. cosine T_max=150.
#   wandb latent-pipeline-nevelT-gn. tsne_n_batches=16 -> live latent_eff_dim (success metric).
#
# PREREQS (all public / already built for ne_vel_T): Ne_norm, T_norm, velocity cache.
#
#   sbatch experiments/ne_vel_T_gn/fm_ne_vel_T_gn.sbatch
#
#SBATCH -J fm_ne_vel_T_gn
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

echo "Node: $(hostname) | FM [nbody,vel] -> [Mgas,ne,T], GROUPNORM (encoder3D parity)"
nvidia-smi

srun python train.py --config experiments/ne_vel_T_gn/fm_ne_vel_T_gn.yaml
