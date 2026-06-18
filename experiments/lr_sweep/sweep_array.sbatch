#!/bin/bash
#SBATCH -J lrsweep
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH -t 1-00:00
#SBATCH --array=0-3
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

module load python
source /mnt/home/mliu1/env/bin/activate
cd /mnt/home/mliu1/latent_pipeline
mkdir -p logs

VARIANTS=(aggr_cosine cosine_restarts onecycle_hi plateau_adaptive)
NAME=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
echo "Node: $(hostname) | variant: $NAME"
nvidia-smi
srun python train.py --config experiments/lr_sweep/$NAME.yaml
