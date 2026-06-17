#!/bin/bash
# Magneticum OOD eval for the TNG-only NF. Auto-picks the best (lowest val/nll)
# checkpoint from nf_ood_test/ckpt and runs nf.ood -> truth-vs-pred OOD plot.
#
#   bash nf_ood_test/eval.sh            # run on a GPU node (or CPU fallback)
# or submit to SLURM:
#   ./submit.sh nf_ood nf_ood_test/config_tng.yaml \
#       --checkpoint <best.ckpt> --tag tng_only --output_dir nf_ood_test/outputs
set -e
cd "$(dirname "$0")/.."          # -> latent_pipeline/

CKPT=$(python - <<'PY'
import glob, os
best = bv = None
for p in glob.glob("nf_ood_test/ckpt/nf-*.ckpt"):
    b = os.path.basename(p)[3:-5]          # strip 'nf-' and '.ckpt'
    try:
        ep, nll = b.split("-", 1)          # split once; nll may start with '-'
        v = float(nll)
    except ValueError:
        continue
    if bv is None or v < bv:                # lowest val/nll wins
        bv, best = v, p
assert best, "no nf-*.ckpt in nf_ood_test/ckpt — train first"
print(best)
PY
)
echo "Best ckpt: $CKPT"
python -m nf.ood --config nf_ood_test/config_tng.yaml \
    --checkpoint "$CKPT" --tag tng_only --output_dir nf_ood_test/outputs
echo "OOD plot -> nf_ood_test/outputs/ood_Magneticum_tng_only.png"
