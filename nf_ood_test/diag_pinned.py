"""Diagnose the prior-wall pile-up in the Swift-EAGLE OOD NF eval.

Flags 'pinned' cubes (posterior mean stuck at the training-prior box edge),
dumps their indices, and makes comparison plots (pinned vs in-manifold) so the
field-level anomaly (if any) is visible. Reads only the predictions npz + the
raw held-out Mgas mmap — no GPU, no model.

  source /mnt/home/mliu1/env/bin/activate
  python nf_ood_test/diag_pinned.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "outputs", "predictions_ood_Swift-EAGLE_real.npz")
MGAS = "/mnt/ceph/users/mliu1/CAMELS-L25n256/Grids_Mgas_Swift-EAGLE_LH_128_z=0.0.npy"
OUT = os.path.join(HERE, "outputs")
# shared norm3d 'gas' stats (training normalisation)
GMEAN, GSTD = 22.003, 1.309
# prior box (training LH ranges)
OM_LO, OM_HI = 0.10, 0.50
S8_LO, S8_HI = 0.60, 1.00
# 'pinned' = posterior mean within this fraction of box width from an edge
EDGE_FRAC = 0.03

d = np.load(PRED)
yt, ym, ys, aux = d["y_true"], d["y_mean"], d["y_std"], d["aux_pred"]
N = len(yt)
om_t, s8_t = yt[:, 0], yt[:, 1]
om_p, s8_p = ym[:, 0], ym[:, 1]

om_w, s8_w = OM_HI - OM_LO, S8_HI - S8_LO
# ANOMALY = high-wall collapse: flow pred stuck near Om upper edge while the
# encoder summary is degenerate (both flow + aux emit a fixed point). Low-wall
# (pred~0.10) cubes are GENUINELY low-Om and correct -> exclude from 'anomaly'.
anom = om_p > OM_HI - EDGE_FRAC * om_w          # high-wall (wrong)
lowwall = om_p < OM_LO + EDGE_FRAC * om_w        # low-wall (mostly correct)
pinned = anom                                     # focus plots on the failure set
normal = ~(anom | lowwall)                        # clean in-manifold
print(f"N={N}  anomaly(high-wall)={anom.sum()}  lowwall(correct)={lowwall.sum()}  "
      f"clean={normal.sum()}")
print(f"  anomaly true-Om: mean={om_t[anom].mean():.3f} span [{om_t[anom].min():.3f},{om_t[anom].max():.3f}]")
print(f"  anomaly aux-Om : mean={aux[anom,0].mean():.3f} std={aux[anom,0].std():.4f}  (constant => encoder degenerate)")

# ---- per-cube field summary stats (cache to npz, recompute if missing) ------
statf = os.path.join(OUT, "eagle_field_stats.npz")
if os.path.exists(statf):
    S = np.load(statf)
    fmean, fstd, fmax, fp99, fhi = S["fmean"], S["fstd"], S["fmax"], S["fp99"], S["fhi"]
else:
    a = np.load(MGAS, mmap_mode="r")
    fmean = np.empty(N); fstd = np.empty(N); fmax = np.empty(N)
    fp99 = np.empty(N); fhi = np.empty(N)
    for i in range(N):
        v = (np.log1p(np.asarray(a[i], dtype=np.float64)) - GMEAN) / GSTD
        fmean[i] = v.mean(); fstd[i] = v.std(); fmax[i] = v.max()
        fp99[i] = np.percentile(v, 99); fhi[i] = (v > 4).mean()
        if i % 100 == 0:
            print(f"  field stats {i}/{N}")
    np.savez(statf, fmean=fmean, fstd=fstd, fmax=fmax, fp99=fp99, fhi=fhi)
    print(f"saved {statf}")

# ---- dump pinned index list -------------------------------------------------
idx = np.arange(N)
listf = os.path.join(OUT, "pinned_indices.txt")
with open(listf, "w") as fh:
    fh.write("# idx om_true s8_true om_pred s8_pred om_std s8_std aux_om aux_s8\n")
    for i in idx[pinned]:
        fh.write(f"{i} {om_t[i]:.4f} {s8_t[i]:.4f} {om_p[i]:.4f} {s8_p[i]:.4f} "
                 f"{ys[i,0]:.4f} {ys[i,1]:.4f} {aux[i,0]:.4f} {aux[i,1]:.4f}\n")
np.savez(os.path.join(OUT, "pinned_mask.npz"),
         anomaly=anom, lowwall=lowwall, clean=normal, idx=idx[pinned])
print(f"saved {listf}  ({pinned.sum()} cubes)")

# ============================ PLOTS ==========================================
def sc(ax, m, c, l):
    ax.scatter(om_t[m] if l != "y" else s8_t[m], None) if False else None

# --- Fig 1: scatter, pinned highlighted -------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for j, (t, p, lab, lo, hi) in enumerate([
        (om_t, om_p, r"$\Omega_m$", OM_LO, OM_HI),
        (s8_t, s8_p, r"$\sigma_8$", S8_LO, S8_HI)]):
    ax[j].scatter(t[normal], p[normal], s=10, alpha=0.4, label="in-manifold", color="tab:blue")
    ax[j].scatter(t[pinned], p[pinned], s=14, alpha=0.7, label="pinned", color="tab:red")
    ax[j].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[j].set_xlabel(f"{lab} true"); ax[j].set_ylabel(f"{lab} pred"); ax[j].set_title(lab)
    ax[j].legend()
fig.suptitle("Swift-EAGLE OOD — pinned (prior-wall) cubes highlighted")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "diag1_scatter.png"), dpi=120); plt.close(fig)

# --- Fig 2: where pinned cubes sit in TRUE (Om,s8) plane ---------------------
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(om_t[normal], s8_t[normal], s=10, alpha=0.4, color="tab:blue", label="in-manifold")
ax.scatter(om_t[pinned], s8_t[pinned], s=18, alpha=0.8, color="tab:red", label="pinned")
ax.set_xlabel(r"$\Omega_m$ true"); ax.set_ylabel(r"$\sigma_8$ true")
ax.set_title("Do pinned cubes cluster in true-param space?"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "diag2_trueplane.png"), dpi=120); plt.close(fig)

# --- Fig 3: field-stat histograms, pinned vs normal --------------------------
stats = [("log1p mean", fmean), ("log1p std", fstd), ("max", fmax),
         ("p99", fp99), ("frac>4", fhi), ("post std Om", ys[:, 0])]
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for ax, (name, arr) in zip(axs.ravel(), stats):
    lo, hi = arr.min(), arr.max()
    bins = np.linspace(lo, hi, 40)
    ax.hist(arr[normal], bins=bins, density=True, alpha=0.5, label="in-manifold", color="tab:blue")
    ax.hist(arr[pinned], bins=bins, density=True, alpha=0.5, label="pinned", color="tab:red")
    ax.set_title(name); ax.legend()
fig.suptitle("Field / posterior stats: pinned vs in-manifold")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "diag3_stats.png"), dpi=120); plt.close(fig)

# --- Fig 4: aux head vs flow mean (does encoder still read right?) -----------
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for j, (t, p, a_, lab) in enumerate([
        (om_t, om_p, aux[:, 0], r"$\Omega_m$"),
        (s8_t, s8_p, aux[:, 1], r"$\sigma_8$")]):
    ax[j].scatter(t[pinned], p[pinned], s=14, alpha=0.6, color="tab:red", label="flow mean")
    ax[j].scatter(t[pinned], a_[pinned], s=14, alpha=0.6, color="tab:green", label="aux head")
    lo = min(t.min(), 0); hi = t.max()
    ax[j].plot([t.min(), t.max()], [t.min(), t.max()], "k--", lw=1)
    ax[j].set_xlabel(f"{lab} true"); ax[j].set_ylabel(f"{lab} pred"); ax[j].set_title(f"{lab}: pinned cubes only")
    ax[j].legend()
fig.suptitle("Pinned cubes — flow collapses, does aux head?")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "diag4_aux.png"), dpi=120); plt.close(fig)

# --- Fig 5: example projected slices, pinned vs normal -----------------------
a = np.load(MGAS, mmap_mode="r")
pin_ex = idx[pinned][:4]; nrm_ex = idx[normal][:4]
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
for col, i in enumerate(pin_ex):
    proj = np.log1p(np.asarray(a[i], dtype=np.float64)).mean(0)
    axs[0, col].imshow(proj, cmap="viridis"); axs[0, col].set_title(f"PINNED #{i}\nOm_t={om_t[i]:.2f} pred={om_p[i]:.2f}")
    axs[0, col].axis("off")
for col, i in enumerate(nrm_ex):
    proj = np.log1p(np.asarray(a[i], dtype=np.float64)).mean(0)
    axs[1, col].imshow(proj, cmap="viridis"); axs[1, col].set_title(f"normal #{i}\nOm_t={om_t[i]:.2f} pred={om_p[i]:.2f}")
    axs[1, col].axis("off")
fig.suptitle("Mgas log1p mean-projection: pinned (top) vs in-manifold (bottom)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "diag5_slices.png"), dpi=120); plt.close(fig)

print("plots -> diag1_scatter diag2_trueplane diag3_stats diag4_aux diag5_slices (.png)")
