#!/bin/bash
# FULL multi-task FM: input [nbody_density, nbody_vel] -> output [Mgas, ne, T]
# (encoder3D parity). The bet to break the latent collapse: gas+ne was NOT enough
# (ne_vel ep86 latent eff-dim 1.00/8, WORSE than ep486 tanh 1.48). Temperature is a
# strong feedback discriminator (AGN/shock heating) -> forces the encoder to use >1 dim;
# velocity should now help via shock/velocity-dispersion -> T.
#
#   in_channels = 5 = x_t(3 = Mgas,ne,T) + nbody(1) + vel(1);  out_channels = 3.
#   Encoder encodes the 3 targets (Mgas,ne,T) -> 8-dim, latent_head=mlp (no tanh), base40.
#   Static input = N-body ONLY; Mgas is generated (flow state), not a static input.
# Warm-start UNet from dim8 ep486 (init_strict=false): enc1 input 2->5 role-remap, out_conv
# 1->3 (ne/T rows zero), encoder FRESH (base40, 3-ch stem). cosine T_max=150, noise_std 0.1.
# wandb latent-pipeline-nevelT. tsne_n_batches=16 -> live latent_eff_dim (the success metric).
#
# PREREQS (all public, NO voxelisation):
#   python prep_cache.py --with_ne      (DONE — Ne_norm_{suite} exist)
#   python prep_cache.py --with_temp    (T_norm_{suite}; log1p + FRESH pool z-score)
#   velocity cache (Grids_Vcdm_Nbody + NbodyVel_norm) — already built for ne_vel.
#
#   sbatch experiments/ne_vel_T/fm_ne_vel_T.sbatch
#
#SBATCH -J fm_ne_vel_T
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

echo "Node: $(hostname) | FM [nbody,vel] -> [Mgas,ne,T], encoder3D-parity multi-task"
nvidia-smi

srun python train.py --config experiments/ne_vel_T/fm_ne_vel_T.yaml
