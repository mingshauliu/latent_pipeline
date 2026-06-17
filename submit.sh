#!/bin/bash
# Thin sbatch wrapper.
#   ./submit.sh train                  # FM train, config/config.yaml
#   ./submit.sh infer                  # FM inference
#   ./submit.sh nf_train [cfg] [over]  # NF train
#   ./submit.sh nf_infer [cfg] [over]  # NF inference
MODE="${1:-train}"
shift
sbatch --job-name="LATENT_${MODE}" run.sh "$MODE" "$@"
