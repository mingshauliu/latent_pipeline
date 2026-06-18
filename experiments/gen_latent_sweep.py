#!/usr/bin/env python
"""Generate the latent-regularization sweep: does a KL term and/or a smaller latent
dim de-degenerate the FM-encoder latent?

Diagnosis (cached/latent_tsne.png, latent_corner.png): the deterministic 8-dim
GasEncoder latent is degenerate (eff dim 1.42/8, off-diag corr up to 0.997) and
disjoint (SIMBA splits off; TNG+Astrid overlap). Three fine-tunes test the fixes:
  - kl_dim8  : beta-VAE encoder (Gaussian z, no tanh), latent_dim=8
  - det_dim3 : deterministic tanh encoder, latent_dim=3
  - kl_dim3  : both — beta-VAE, latent_dim=3

Each WARM-STARTS (weights only, fresh optimiser+scheduler, epoch 0) from the
converged sv6mp9wt best-149 (val 0.0154) via train.py init_from + init_strict=False
(the latent head / FiLM-fusion layers change shape and are smart-initialised). A
moderate cosine anneal (lr 1e-4 -> 1e-6 over 150 ep) settles the warm-started
weights into the new latent geometry. Single-GPU each so 3 variants fit one alloc.
"""
import copy, os, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "latent_sweep")
os.makedirs(OUT, exist_ok=True)

# converged plateau checkpoint (val 0.0154, EMA baked into state_dict)
INIT = "latent-pipeline/sv6mp9wt/checkpoints/best-epoch=149-val_loss=0.015439.ckpt"

with open(os.path.join(ROOT, "config", "config.yaml")) as f:
    base = yaml.safe_load(f)

# shared training overrides: warm-start + moderate cosine anneal (150 ep, 1e-4->1e-6)
COMMON_TRAIN = dict(
    init_from=INIT,        # weights-only warm start (train.py); NOT resume_from
    init_strict=False,     # partial load: latent head / cond_fuse reshaped
    resume_from=None,
    devices=1,             # one GPU per variant -> 3 fit one alloc
    strategy="auto",
    scheduler="cosine",
    lr=1.0e-4,             # below the 6e-4 from-scratch lr (weights are converged)
    warmup_epochs=5,
    cosine_t_max=150,      # anneal over the whole run
    eta_min=1.0e-6,
    max_epochs=150,
    wandb_project="latent-pipeline-latentsweep",
)

KL = dict(kl_beta=1.0e-3, kl_warmup_epochs=30, kl_free_bits=0.5)

# variant -> (model overrides, extra training overrides)
VARIANTS = {
    # H1: KL prior regularises the latent toward a smooth N(0,I) -> less disjoint,
    #     dims self-prune. latent_dim kept at 8 (cond_fuse shape unchanged).
    "kl_dim8":  (dict(variational=True,  latent_dim=8), dict(KL)),
    # H2: shrink to ~eff-dim so dead/degenerate dims can't exist. Deterministic.
    "det_dim3": (dict(variational=False, latent_dim=3), {}),
    # H3: both — small KL-regularised latent.
    "kl_dim3":  (dict(variational=True,  latent_dim=3), dict(KL)),
}

for name, (model_ov, train_ov) in VARIANTS.items():
    cfg = copy.deepcopy(base)
    cfg["model"].update(model_ov)
    cfg["training"].update(COMMON_TRAIN)
    cfg["training"].update(train_ov)
    path = os.path.join(OUT, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {path}  (variational={model_ov['variational']}, "
          f"latent_dim={model_ov['latent_dim']}, max_epochs={COMMON_TRAIN['max_epochs']})")

# SLURM job array: one single-GPU job per variant.
# User submits (FI policy: Claude never runs sbatch):  sbatch sweep_array.sbatch
names = list(VARIANTS)
arr = os.path.join(OUT, "sweep_array.sbatch")
with open(arr, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH -J latentsweep\n#SBATCH -p gpu\n#SBATCH -N 1\n")
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
    f.write('srun python train.py --config experiments/latent_sweep/$NAME.yaml\n')
print(f"wrote {arr}")
