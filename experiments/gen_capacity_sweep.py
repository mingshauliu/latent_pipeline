#!/usr/bin/env python
"""Generate the capacity x bottleneck grid: does a smaller latent_dim and/or a
smaller encoder de-degenerate the FM-encoder latent (eff dim 1.42/8)?

Root-cause hypotheses (user):
  1. single output channel (gas only) -> little independent structure to encode,
     latent collapses to ~1 dim.
  2. encoder over-capacity (SE-ResNet3D base=16: 16->32->64->128) memorises /
     over-compresses (suite-ID, dead dims) instead of a smooth feedback manifold.

Two levers, both config-only (knobs already wired in module.py/model.py):
  - model.latent_dim  : shrink the bottleneck (3 -> 2 / 1).
  - model.encoder_base: shrink the encoder (16 -> 8).

Grid = {latent_dims} x {encoder_bases}. Each WARM-STARTS (weights only, fresh
optimiser, epoch 0) from the det3 sweep winner (vmve93wg best ep008, val 0.014682)
via train.py init_from + init_strict=False (warm_load_partial: copies matching
tensors, smart-inits gas_encoder.proj + net.cond_fuse.0; for base=8 the encoder
convs differ -> encoder retrains fresh, UNet backbone carried). Aggressive cosine
anneal (lr 1e-4 -> 1e-6 over 40 ep, 50 ep total) so it SETTLES instead of bouncing
in the ~0.018 noise band the constant-lr det3 run showed.

Usage:
  python experiments/gen_capacity_sweep.py                 # dims {1,2} x base {16,8}
  python experiments/gen_capacity_sweep.py --latent_dims 2 3   # if det3 eff_dim ~2-2.5
  python experiments/gen_capacity_sweep.py --encoder_bases 16 8 4
Pick --latent_dims from the det3 corner plot eff-dim (experiments/extract_det3.sbatch):
  eff_dim <= ~1.5 -> {1,2} ;  eff_dim ~2-2.5 -> {2,3}.
"""
import argparse, copy, os, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "cap_sweep")

# det3 sweep winner: lowest-val ckpt (ep008 0.014682); ep021/026 drifted up.
# Swap to a det3_anneal best if that run finishes lower than 0.014682.
INIT = ("latent-pipeline-latentsweep/vmve93wg/checkpoints/"
        "best-epoch=008-val_loss=0.014682.ckpt")

# warm-start + aggressive anneal (the det3_anneal schedule that settles).
COMMON_TRAIN = dict(
    init_from=INIT,        # weights-only warm start (train.py); NOT resume_from
    init_strict=False,     # partial load: latent head / cond_fuse (+ encoder if base!=16)
    resume_from=None,
    devices=1,             # one GPU per variant
    strategy="auto",
    scheduler="cosine",
    lr=1.0e-4,
    warmup_epochs=3,
    cosine_t_max=40,       # hard anneal -> lr reaches eta_min, settles
    eta_min=1.0e-6,
    max_epochs=50,
    batch_size=2,          # UNet (base_channels=128 @128^3) is the memory hog -> keep 2
    wandb_project="latent-pipeline-capsweep",
)

# A smaller encoder frees headroom AND the batch=2 grad noise is what made det3
# bounce in the ~0.018 band. Raise the EFFECTIVE batch via accumulate_grad (per-step
# memory unchanged -> no OOM risk) inversely with encoder_base. base16 keeps the
# det3 eff-batch 8; base8 -> 16; base4 -> 32. Less grad noise => settles lower.
REF_BASE = 16
REF_ACCUM = 4              # det3 used batch2 x accum4 = eff batch 8


def accum_for(b):
    return REF_ACCUM * max(REF_BASE // b, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dims", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--encoder_bases", type=int, nargs="+", default=[16, 8])
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(ROOT, "config", "config.yaml")) as f:
        base = yaml.safe_load(f)

    names = []
    for d in args.latent_dims:
        for b in args.encoder_bases:
            name = f"dim{d}_base{b}"
            cfg = copy.deepcopy(base)
            cfg["model"].update(dict(variational=False, latent_dim=d, encoder_base=b))
            cfg["training"].update(COMMON_TRAIN)
            accum = accum_for(b)
            cfg["training"]["accumulate_grad"] = accum
            path = os.path.join(OUT, f"{name}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            names.append(name)
            print(f"wrote {path}  (latent_dim={d}, encoder_base={b}, "
                  f"accum={accum} -> eff_batch={2 * accum})")

    arr = os.path.join(OUT, "sweep_array.sbatch")
    with open(arr, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -J capsweep\n#SBATCH -p gpu\n#SBATCH -N 1\n")
        f.write("#SBATCH --gres=gpu:1\n#SBATCH --constraint=a100-80gb\n")
        f.write("#SBATCH --ntasks-per-node=1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=480G\n")
        f.write("#SBATCH -t 1-12:00\n")
        f.write(f"#SBATCH --array=0-{len(names) - 1}\n")
        f.write("#SBATCH -o logs/%x_%A_%a.out\n#SBATCH -e logs/%x_%A_%a.err\n\n")
        f.write("module load python\nsource /mnt/home/mliu1/env/bin/activate\n")
        f.write(f"cd {ROOT}\nmkdir -p logs\n\n")
        f.write("VARIANTS=(%s)\n" % " ".join(names))
        f.write('NAME=${VARIANTS[$SLURM_ARRAY_TASK_ID]}\n')
        f.write('echo "Node: $(hostname) | variant: $NAME"\nnvidia-smi\n')
        f.write('srun python train.py --config experiments/cap_sweep/$NAME.yaml\n')
    print(f"wrote {arr}  ({len(names)}-job array)")


if __name__ == "__main__":
    main()
