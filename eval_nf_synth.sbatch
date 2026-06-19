#!/bin/bash
# real-NF truth-vs-pred on FM-synth Mgas (dim8). Fast synth-fidelity judge: does
# dim8 FM-synth retain the per-cube (Om, s8) signal that real-NF reads off real Mgas?
# Baseline (real Mgas, plot_nf_2param --data_mode real): Om R2=0.966, s8 R2=0.886.
# Runs BOTH oracle (encode) and deploy (mean) latent modes.
#   sbatch eval_nf_synth.sbatch
#
#SBATCH -J nf_synth
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH -t 0-02:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs outputs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FM=latent-pipeline/sv6mp9wt/checkpoints/best-epoch=269-val_loss=0.013353.ckpt
NF=nf_ck_real/nf-752--5.4560.ckpt
echo "Node: $(hostname) | FM: $FM | NF: $NF"
nvidia-smi

for MODE in encode mean; do
  echo "===== latent_mode=$MODE ====="
  python eval_nf_synth.py \
      --config config/config.yaml \
      --fm_checkpoint "$FM" \
      --nf_checkpoint "$NF" \
      --latent_mode "$MODE" \
      --noise_std 0.1 \
      --max_cubes 300 --num_steps 50 --method dopri5
done
