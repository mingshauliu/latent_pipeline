#!/bin/bash
# Four truth-vs-pred plots (TNG in-dist + Astrid/SIMBA/Magneticum OOD) for the
# IllustrisTNG-only NF, on 1 GPU. Submit from latent_pipeline/:
#   sbatch nf_ood_test/eval_four.sh
# Optional args forwarded to eval_four.py, e.g.:
#   sbatch nf_ood_test/eval_four.sh --n_lh 300 --checkpoint nf_ood_test/ckpt/last.ckpt

#SBATCH -J LATENT_eval4
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-01:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
cd /mnt/home/mliu1/latent_pipeline
mkdir -p logs nf_ood_test/outputs

echo "Node: $(hostname)"
nvidia-smi

# Defaults: best ckpt auto-picked, 200 cubes/LH suite, all 48 Magneticum, 2000 draws.
python nf_ood_test/eval_four.py --device cuda --n_lh 200 --n_post 2000 "$@"

echo "Done. Plots -> nf_ood_test/outputs/four_<suite>.png"
