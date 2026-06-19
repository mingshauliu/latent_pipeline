"""In-distribution sanity for the det3 FM (latent_dim=3): generate synth Mgas for a
few training-suite cubes and show (a) imshow Nbody|true|synth|residual and (b)
truth-vs-pred (voxel 2D hist + mean power spectrum + xcorr/pk numbers).

latent_mode=encode -> conditions on the latent encoded from the TRUE Mgas (the
oracle / in-distribution mode training's xcorr used). This is the model's best-case
field fidelity, not the deploy-time marginal (latent_mode=mean).

    python eval_indist.py --config experiments/latent_sweep/det_dim3.yaml \
        --checkpoint latent-pipeline-latentsweep/vmve93wg/checkpoints/best-epoch=008-val_loss=0.014682.ckpt
GPU (FM dopri5 sampling at 128^3). Writes outputs/indist_<suite>_imshow.png + _tvp.png.
"""
import argparse, os
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module import FlowMatchingModel, xcorr_metric
from infer import load_norm, norm_field, norm_nbody, nbody_div

try:
    import Pk_library as PKL
    HAS_PKL = True
except Exception:
    HAS_PKL = False


def pk_1d(field, box):
    """P(k) of the demeaned NORMED log-Mgas field (the model's native space).
    field is log1p+z-score (~zero mean) -> use delta = f - mean as the contrast.
    Everything stays in log space: NO expm1 to physical (in-distribution compare)."""
    d = (field - field.mean()).astype(np.float32)
    Pk = PKL.Pk(np.ascontiguousarray(d), box, axis=0, MAS="CIC", threads=1)
    return Pk.k3D, Pk.Pk[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/latent_sweep/det_dim3.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default=None, help="source name substring; default=first source")
    ap.add_argument("--n", type=int, default=4, help="cubes to generate")
    ap.add_argument("--latent_mode", default="encode", choices=["encode", "mean"])
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--method", default="dopri5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dev = args.device if torch.cuda.is_available() else "cpu"
    norm = load_norm()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.checkpoint}")
    model = FlowMatchingModel.load_from_checkpoint(
        args.checkpoint, cfg=cfg, strict=False).to(dev).eval()
    ldim = cfg["model"].get("latent_dim", 8)
    box = cfg["data"]["box_size"]
    print(f"latent_dim={ldim} | box={box} | mode={args.latent_mode} | dev={dev}")

    srcs = cfg["inference"]["sources"]
    src = next((s for s in srcs if args.suite and args.suite in s["name"]), srcs[0])
    suite = src["name"]
    print(f"source: {suite}")
    nbody = np.load(src["nbody_path"], mmap_mode="r")
    mgas = np.load(src["mgas_path"], mmap_mode="r")
    cosmo_all = np.loadtxt(src["param_path"]).astype(np.float32)[:, :cfg["data"].get("n_cosmo", 2)]

    N = min(args.n, len(nbody))
    div = nbody_div(suite, cfg)
    clamp_val = cfg["data"].get("clamp_val", 10.0)
    # Everything in NORMED LOG space (model native), and clamped to +/-clamp_val
    # EXACTLY as data.CachedFMDataset does in training: nb = scalar-corrected norm3d
    # log-mass Nbody, true = log1p+gas-zscore Mgas, synth = raw output (NO expm1).
    nb_raw, true_raw, synth_raw = [], [], []
    for i in range(N):
        nb = norm_nbody(np.asarray(nbody[i], dtype=np.float32),
                        norm["nbody_mean"], norm["nbody_std"], div=div)
        true_n = norm_field(np.asarray(mgas[i], dtype=np.float32), norm["mgas_mean"], norm["mgas_std"])
        if clamp_val is not None:
            np.clip(nb, -clamp_val, clamp_val, out=nb)
            np.clip(true_n, -clamp_val, clamp_val, out=true_n)
        nb_t = torch.from_numpy(nb)[None, None].to(dev)
        cosmo = (cosmo_all[i] - norm["cosmo_mean"]) / norm["cosmo_std"]
        cosmo_t = torch.from_numpy(cosmo.astype(np.float32))[None].to(dev)

        if args.latent_mode == "encode":
            with torch.no_grad():
                enc = model.gas_encoder(torch.from_numpy(true_n)[None, None].to(dev))
            latent = enc[0] if isinstance(enc, tuple) else enc
        else:
            latent = torch.zeros(1, ldim, device=dev)

        with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            s = model.sample(nb_t, cosmo_t, latent, num_steps=args.num_steps, method=args.method)
        synth_n = s[0, 0].float().cpu().numpy()   # stays in normed log space
        nb_raw.append(nb)
        true_raw.append(true_n)
        synth_raw.append(synth_n)
        xc = xcorr_metric(synth_n, true_n, box) if HAS_PKL else float("nan")
        print(f"  cube {i}: xcorr={xc:.4f}  true(norm)[{true_n.min():.2f},{true_n.max():.2f}]  "
              f"synth(norm)[{synth_n.min():.2f},{synth_n.max():.2f}]")

    # ── (a) imshow: per-cube mid-slice, all in NORMED LOG space ──────────────────
    z = nb_raw[0].shape[0] // 2
    fig, ax = plt.subplots(N, 4, figsize=(13, 3.1 * N))
    ax = np.atleast_2d(ax)
    cols = ["Nbody (norm)", "true Mgas (norm log)",
            f"synth Mgas (norm log, {args.latent_mode})", "synth − true"]
    for i in range(N):
        t_s, s_s = true_raw[i][z], synth_raw[i][z]
        vmin, vmax = min(t_s.min(), s_s.min()), max(t_s.max(), s_s.max())
        ax[i, 0].imshow(nb_raw[i][z], cmap="viridis")
        ax[i, 1].imshow(t_s, cmap="Blues", vmin=vmin, vmax=vmax)
        ax[i, 2].imshow(s_s, cmap="Blues", vmin=vmin, vmax=vmax)
        d = ax[i, 3].imshow(s_s - t_s, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(d, ax=ax[i, 3], fraction=0.046)
        ax[i, 0].set_ylabel(f"cube {i}")
    for j, c in enumerate(cols):
        ax[0, j].set_title(c, fontsize=10)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{suite} in-distribution (norm log space) — det3 FM (latent_dim={ldim}), slice z={z}")
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, f"indist_{suite}_imshow.png")
    fig.savefig(p1, dpi=130); print(f"\nSaved {p1}")

    # ── (b) truth-vs-pred: voxel 2D hist + mean P(k) + ratio — all NORMED LOG ─────
    tv = np.concatenate([t.ravel() for t in true_raw])
    sv = np.concatenate([s.ravel() for s in synth_raw])
    fig2, ax2 = plt.subplots(1, 3 if HAS_PKL else 1, figsize=(14 if HAS_PKL else 5, 4.2))
    ax2 = np.atleast_1d(ax2)
    lo, hi = min(tv.min(), sv.min()), max(tv.max(), sv.max())
    h = ax2[0].hist2d(tv, sv, bins=200, range=[[lo, hi], [lo, hi]], cmap="inferno",
                      norm=matplotlib.colors.LogNorm())
    ax2[0].plot([lo, hi], [lo, hi], "w--", lw=1)
    plt.colorbar(h[3], ax=ax2[0], fraction=0.046)
    ax2[0].set_xlabel("true (norm log Mgas)"); ax2[0].set_ylabel("synth (norm log Mgas)")
    ax2[0].set_title(f"voxel truth-vs-pred ({N} cubes)")

    if HAS_PKL:
        ks, pts, pss, rats, xcs = None, [], [], [], []
        for i in range(N):
            k, pt = pk_1d(true_raw[i], box)     # demeaned normed-log field
            _, ps = pk_1d(synth_raw[i], box)
            ks = k; pts.append(pt); pss.append(ps); rats.append(ps / (pt + 1e-30))
            xcs.append(xcorr_metric(synth_raw[i], true_raw[i], box))
        pt_m, ps_m, r_m = np.mean(pts, 0), np.mean(pss, 0), np.mean(rats, 0)
        ax2[1].loglog(ks, pt_m, "k-", label="true")
        ax2[1].loglog(ks, ps_m, "r--", label="synth")
        ax2[1].axvline(15, color="gray", ls=":", lw=1)
        ax2[1].set_xlabel("k [h/Mpc]"); ax2[1].set_ylabel("P(k) of norm-log Mgas")
        ax2[1].legend(); ax2[1].set_title("mean power spectrum (log space)")
        ax2[2].semilogx(ks, r_m, "b-")
        ax2[2].axhline(1, color="k", lw=0.8)
        ax2[2].axhspan(0.9, 1.1, color="g", alpha=0.12)
        ax2[2].axvline(15, color="gray", ls=":", lw=1)
        ax2[2].set_ylim(0.5, 1.5); ax2[2].set_xlabel("k [h/Mpc]")
        ax2[2].set_ylabel("P_synth / P_true"); ax2[2].set_title("P(k) ratio (±10% band)")
        fig2.suptitle(f"{suite} det3 in-dist (norm log) — mean xcorr={np.mean(xcs):.4f}", fontsize=11)
    fig2.tight_layout()
    p2 = os.path.join(args.out_dir, f"indist_{suite}_tvp.png")
    fig2.savefig(p2, dpi=130); print(f"Saved {p2}")
    if HAS_PKL:
        print(f"\nmean xcorr={np.mean(xcs):.4f} | P(k) ratio @k<15 mean={np.mean(r_m[ks<=15]):.4f}")


if __name__ == "__main__":
    main()
