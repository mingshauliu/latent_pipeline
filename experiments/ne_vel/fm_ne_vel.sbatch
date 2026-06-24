#!/bin/bash
# COMBINED FM: input [nbody, nbody_vel] -> output [Mgas, ne], with an encoder3D-sized
# latent encoder. The bet: same param budget as encoder3D (ResNetBranch 8.27M; our
# GasEncoder base=40 -> 9.18M) BUT with our better learning schedule + richer 2-ch
# (Mgas,ne) encode target + velocity conditioning -> a BETTER, NON-COLLAPSED latent
# (ep486 deterministic latent eff-dim was only 1.48/8).
#
# Mechanics (all gated, see CLAUDE.md):
#   use_velocity -> N-body VELOCITY input channel ([x_t, nbody, vel]); high-k flow info.
#   use_ne       -> ne 2nd OUTPUT target (out_channels=2) + 2-ch (Mgas,ne) encoder input.
#   in_channels = 4 = x_t(2) + nbody(1) + vel(1).
# Warm-start UNet backbone from dim8 best ep486 (init_strict=false): enc1 input 2->4
# role-remapped (Mgas x_t + nbody kept; ne x_t + vel fresh), out_conv 1->2 (ne row zero),
# encoder FRESH (base 16->40). Fresh optimiser, cosine T_max=150 lr 2e-4->1e-6 / 150 ep,
# noise_std 0.1. wandb latent-pipeline-nevel.
#
# PREREQS:
#   - ne cache: `python prep_cache.py --with_ne`            (DONE — Ne_norm_{suite} exist)
#   - velocity cache: after the NbodyVel voxelisation finishes ALL 3 suites,
#       python voxelise/voxelise_nbody_vel.py --suite all --stack   -> Grids_Vcdm_Nbody
#       python prep_cache.py --with_vel                             -> NbodyVel_norm + stats
#     (as of 2026-06-23 SIMBA NbodyVel still voxelising -> velocity cache PENDING.)
#
#   sbatch experiments/ne_vel/fm_ne_vel.sbatch
#
#SBATCH -J fm_ne_vel
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

echo "Node: $(hostname) | FM [nbody,nbody_vel] -> [Mgas,ne], encoder3D-sized encoder"
nvidia-smi

srun python train.py --config experiments/ne_vel/fm_ne_vel.yaml
