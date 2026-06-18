"""GPU: dump NF encoder summary vectors for the Swift-EAGLE held-out pool (and,
optionally, a training-suite pool) to locate the encoder collapse behind the
prior-wall pile-up.

The CPU diag (diag_pinned.py) showed: ~150 EAGLE cubes get a posterior mean
stuck at the Om upper wall AND an aux-head output that is EXACTLY constant
(std=0). That means the encoder summary is degenerate/saturated for those cubes.
This script extracts the actual summary (summary_dim) per cube so we can see
which latent dim saturates and whether the anomaly cubes occupy an off-manifold
region vs the training pool.

  source /mnt/home/mliu1/env/bin/activate
  python nf_ood_test/dump_summaries.py \
      --checkpoint nf_ck_real/nf-752--5.4560.ckpt \
      --mgas /mnt/ceph/users/mliu1/CAMELS-L25n256/Grids_Mgas_Swift-EAGLE_LH_128_z=0.0.npy

Writes nf_ood_test/outputs/eagle_summaries.npz (summary, aux) and
diag6_summaries.png (per-dim summary, anomaly vs clean).
"""
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nf.predict import load_nf                       # noqa: E402

GMEAN, GSTD = 22.003, 1.309
OUT = os.path.join(HERE, "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mgas",
                    default="/mnt/ceph/users/mliu1/CAMELS-L25n256/Grids_Mgas_Swift-EAGLE_LH_128_z=0.0.npy")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    model = load_nf(args.checkpoint, device=dev)

    a = np.load(args.mgas, mmap_mode="r")
    N = len(a)
    summ, auxl = [], []
    with torch.no_grad():
        for s in range(0, N, args.batch):
            e = min(N, s + args.batch)
            chunk = np.stack([(np.log1p(np.asarray(a[i], dtype=np.float32)) - GMEAN) / GSTD
                              for i in range(s, e)])
            x = torch.from_numpy(chunk).unsqueeze(1).to(dev)   # (B,1,128,128,128)
            summary, aux = model(x)
            summ.append(summary.cpu().numpy())
            auxl.append(aux.cpu().numpy())
            if s % 80 == 0:
                print(f"  {s}/{N}")
    summary = np.concatenate(summ); aux = np.concatenate(auxl)
    np.savez(os.path.join(OUT, "eagle_summaries.npz"), summary=summary, aux=aux)
    print(f"summary shape {summary.shape}  saved eagle_summaries.npz")

    # anomaly mask from predictions (high-wall) if available
    predf = os.path.join(OUT, "predictions_ood_Swift-EAGLE_real.npz")
    if os.path.exists(predf):
        d = np.load(predf)
        om_p = d["y_mean"][:, 0]
        anom = om_p > 0.485
    else:
        anom = np.zeros(N, dtype=bool)
    clean = ~anom

    D = summary.shape[1]
    fig, axs = plt.subplots(2, (D + 1) // 2, figsize=(4 * ((D + 1) // 2), 8))
    for k, ax in enumerate(axs.ravel()):
        if k >= D:
            ax.axis("off"); continue
        lo, hi = summary[:, k].min(), summary[:, k].max()
        bins = np.linspace(lo, hi, 40)
        ax.hist(summary[clean, k], bins=bins, density=True, alpha=0.5, color="tab:blue", label="clean")
        ax.hist(summary[anom, k], bins=bins, density=True, alpha=0.5, color="tab:red", label="anomaly")
        ax.set_title(f"summary dim {k}  (anom std={summary[anom,k].std():.3f})")
        ax.legend()
    fig.suptitle("NF encoder summary per dim — anomaly (high-wall) vs clean EAGLE cubes")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "diag6_summaries.png"), dpi=120)
    print("plot -> diag6_summaries.png")

    # which dims are degenerate within the anomaly set?
    print("anomaly-set per-dim std (small => saturated):")
    for k in range(D):
        print(f"  dim {k}: anom std {summary[anom,k].std():.4f}  clean std {summary[clean,k].std():.4f}")


if __name__ == "__main__":
    main()
