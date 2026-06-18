#!/usr/bin/env python
"""Generate LR-schedule sweep configs to test whether the FM val-loss plateau
(~0.0154, sv6mp9wt) is a high-lr noise floor that an aggressive schedule breaks.

Current prod schedule: cosine with T_max=1985 over 2000 epochs -> lr stays ~6e-4
(near peak) for hundreds of epochs and NEVER meaningfully anneals. sv6mp9wt
plateaued at val 0.0154 (epoch 149) and drifted UP after (0.0167 @ 219) at
near-constant lr -> classic too-flat-lr stall.

Each variant WARM-STARTS (weights only, fresh optimiser+scheduler, epoch 0) from
the converged plateau ckpt, so we isolate the schedule effect. Single-GPU each
(devices=1) so 4 variants fit one 4-GPU disBatch allocation.
"""
import copy, os, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "lr_sweep")
os.makedirs(OUT, exist_ok=True)

# converged plateau checkpoint (val 0.0154, EMA baked into state_dict)
INIT = "latent-pipeline/sv6mp9wt/checkpoints/best-epoch=149-val_loss=0.015439.ckpt"

with open(os.path.join(ROOT, "config", "config.yaml")) as f:
    base = yaml.safe_load(f)

COMMON = dict(
    init_from=INIT,     # weights-only warm start (train.py); NOT resume_from
    resume_from=None,
    devices=1,          # one GPU per variant -> 4 fit a 4-GPU alloc via disBatch
    strategy="auto",
    eta_min=1.0e-6,
)

# variant -> training-block overrides
VARIANTS = {
    # H1: plateau is a high-lr noise floor. Drive lr 6e-4 -> 1e-6 hard over 300 ep.
    "aggr_cosine": dict(scheduler="cosine", lr=6.0e-4, warmup_epochs=5,
                        cosine_t_max=300, max_epochs=305),
    # H2: stuck in a poor basin. SGDR re-heats to escape, anneals each cycle.
    #     T_0=50,T_mult=2 -> cycles end at 50,150,350.
    "cosine_restarts": dict(scheduler="cosine_restarts", lr=6.0e-4, warmup_epochs=0,
                            restart_t0=50, restart_tmult=2, max_epochs=350),
    # H3: need a bigger kick than 6e-4 to leave the basin, then hard anneal.
    #     Higher peak -> rely on clamp_val=10 + loss_spike_thresh guard (already on).
    "onecycle_hi": dict(scheduler="onecycle", lr=6.0e-4, warmup_epochs=0,
                        onecycle_max_lr=1.2e-3, onecycle_pct_start=0.1,
                        onecycle_div_factor=10.0, onecycle_final_div=1.0e4,
                        max_epochs=300),
    # H4: adaptive — only drop lr when val actually stalls (control / safe bet).
    "plateau_adaptive": dict(scheduler="plateau", lr=6.0e-4, warmup_epochs=5,
                             plateau_factor=0.5, plateau_patience=15,
                             max_epochs=400),
}

for name, ov in VARIANTS.items():
    cfg = copy.deepcopy(base)
    cfg["training"].update(COMMON)
    cfg["training"].update(ov)
    cfg["training"]["wandb_project"] = "latent-pipeline-lrsweep"
    path = os.path.join(OUT, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {path}  ({ov['scheduler']}, max_epochs={ov['max_epochs']})")

# SLURM job array: one single-GPU job per variant (each gets its own A100).
# User submits (FI policy: Claude never runs sbatch):  sbatch sweep_array.sbatch
names = list(VARIANTS)
arr = os.path.join(OUT, "sweep_array.sbatch")
with open(arr, "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"#SBATCH -J lrsweep\n#SBATCH -p gpu\n#SBATCH -N 1\n")
    f.write("#SBATCH --gres=gpu:1\n#SBATCH --constraint=a100-80gb\n")
    f.write("#SBATCH --ntasks-per-node=1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=480G\n")
    f.write("#SBATCH -t 1-00:00\n")
    f.write(f"#SBATCH --array=0-{len(names) - 1}\n")
    f.write("#SBATCH -o logs/%x_%A_%a.out\n#SBATCH -e logs/%x_%A_%a.err\n\n")
    f.write("module load python\nsource /mnt/home/mliu1/env/bin/activate\n")
    f.write(f"cd {ROOT}\nmkdir -p logs\n\n")
    f.write("VARIANTS=(%s)\n" % " ".join(names))
    f.write('NAME=${VARIANTS[$SLURM_ARRAY_TASK_ID]}\n')
    f.write('echo "Node: $(hostname) | variant: $NAME"\nnvidia-smi\n')
    f.write('srun python train.py --config experiments/lr_sweep/$NAME.yaml\n')
print(f"wrote {arr}")
