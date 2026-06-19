"""Extract FM encoder latents over the training pool, fit a Gaussian envelope.

Step 1+2 of the Gaussian-latent synth plan. The FM conditions on an 8-dim
tanh-bounded latent encoded from Mgas (`GasEncoder`). At sampling time Mgas is
unknown, so we model the latent's *training* distribution and draw from it,
instead of the mismatched N(0,I) (`infer.py` latent_mode=sample) or the
feedback-washing latent=0 (mode=mean).

This script:
  1. Loads the FM checkpoint (EMA weights baked into state_dict by
     module.on_save_checkpoint) -> frozen gas_encoder.
  2. Encodes the PRE-NORMALISED training Mgas cache (log1p+norm3d-gas, exactly
     what the encoder saw in training) for all suites -> Z (N, latent_dim).
  3. Fits a full Gaussian N(mu, Sigma), reports the correlation matrix (settles
     diagonal-vs-full), saves cached/latent_stats.npz, and writes a corner plot.

    python extract_latents.py --checkpoint <fm.ckpt>
"""

import argparse
import os
import numpy as np
import torch
import yaml

from module import FlowMatchingModel

CACHE_TMPL = ("/mnt/ceph/users/mliu1/latent_pipeline_cache/L25_LH_128_norm/"
              "Mgas_norm_{suite}_LH_128_z=0.0.npy")
OUT_NPZ = "cached/latent_stats.npz"
OUT_PNG = "cached/latent_corner.png"
DEFAULT_CKPT = ("latent-pipeline/sv6mp9wt/checkpoints/"
                "best-epoch=149-val_loss=0.015439.ckpt")


@torch.no_grad()
def encode_suite(model, path, dev, batch=8, log_every=200):
    arr = np.load(path, mmap_mode="r")
    n = len(arr)
    out = np.empty((n, model.gas_encoder.proj.out_features), dtype=np.float32)
    for i in range(0, n, batch):
        chunk = np.asarray(arr[i:i + batch], dtype=np.float32)
        x = torch.from_numpy(chunk)[:, None].to(dev)
        with torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            z = model.gas_encoder(x)
        out[i:i + batch] = z.float().cpu().numpy()
        if (i % log_every) < batch:
            print(f"    {os.path.basename(path)}: {min(i+batch,n)}/{n}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--plot_out", default=None,
                    help="corner PNG path; default = --out with _corner.png suffix")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()
    png_out = args.plot_out or (os.path.splitext(args.out)[0] + "_corner.png")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    suites = cfg["data"]["suites"]
    dev = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.checkpoint}")
    model = FlowMatchingModel.load_from_checkpoint(
        args.checkpoint, cfg=cfg, strict=False).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ldim = model.gas_encoder.proj.out_features
    print(f"latent_dim = {ldim} | suites = {suites} | device = {dev}")

    Z_parts, ranges, off = [], {}, 0
    for s in suites:
        path = CACHE_TMPL.format(suite=s)
        print(f"  encoding {s}")
        z = encode_suite(model, path, dev, batch=args.batch)
        Z_parts.append(z)
        ranges[s] = (off, off + len(z)); off += len(z)
    Z = np.concatenate(Z_parts, 0).astype(np.float32)
    print(f"\nZ shape {Z.shape}  min {Z.min():.3f} max {Z.max():.3f}")

    mean = Z.mean(0)
    cov = np.cov(Z, rowvar=False)
    std = Z.std(0)
    corr = np.corrcoef(Z, rowvar=False)
    np.set_printoptions(precision=3, suppress=True)
    print("\nper-dim mean:", mean)
    print("per-dim std :", std)
    print("\ncorrelation matrix (off-diagonal ~0 => diagonal Gaussian suffices):")
    print(corr)
    offdiag = corr[~np.eye(ldim, dtype=bool)]
    print(f"\nmax |off-diagonal corr| = {np.abs(offdiag).max():.3f}  "
          f"mean = {np.abs(offdiag).mean():.3f}")

    # degeneracy: eigen-spectrum of the covariance. Few dominant eigenvalues =>
    # latent lives on a lower-dim subspace (degenerate); full Sigma captures it,
    # a diagonal Gaussian would WRONGLY fill the degenerate directions.
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    var_frac = evals / evals.sum()
    eff_dim = (evals.sum() ** 2) / (evals ** 2).sum()  # participation ratio
    print(f"\ncov eigenvalues: {evals}")
    print(f"variance fraction: {var_frac}")
    print(f"effective dim (participation ratio) = {eff_dim:.2f} / {ldim}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out,
             Z=Z, mean=mean.astype(np.float32), cov=cov.astype(np.float32),
             std=std.astype(np.float32), corr=corr.astype(np.float32),
             suites=np.array(suites),
             ranges=np.array([ranges[s] for s in suites]),
             latent_dim=np.int64(ldim),
             checkpoint=np.array(args.checkpoint))
    print(f"\nSaved {args.out}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse

            def cov_ellipse(ax, mj, mi, c2x2, nsig=1.0, **kw):
                """1-sigma contour of the 2D marginal Gaussian (Mahalanobis=nsig)."""
                vals, vecs = np.linalg.eigh(c2x2)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                w, h = 2 * nsig * np.sqrt(np.maximum(vals, 0))
                e = Ellipse((mj, mi), w, h, angle=ang, fill=False, **kw)
                ax.add_patch(e)

            d = ldim
            fig, ax = plt.subplots(d, d, figsize=(1.6 * d, 1.6 * d))
            for i in range(d):
                for j in range(d):
                    a = ax[i, j]
                    if i == j:
                        a.hist(Z[:, i], bins=40, density=True, color="steelblue")
                        # fitted 1D Gaussian + 1-sigma band
                        xs = np.linspace(Z[:, i].min(), Z[:, i].max(), 200)
                        a.plot(xs, np.exp(-0.5 * ((xs - mean[i]) / std[i]) ** 2)
                               / (std[i] * np.sqrt(2 * np.pi)), "r-", lw=1.2)
                        a.axvspan(mean[i] - std[i], mean[i] + std[i],
                                  color="r", alpha=0.12)
                    else:
                        a.scatter(Z[:, j], Z[:, i], s=2, alpha=0.15, color="k")
                        sub = cov[np.ix_([j, i], [j, i])]
                        cov_ellipse(a, mean[j], mean[i], sub, nsig=1.0,
                                    edgecolor="r", lw=1.4)
                    if i == d - 1: a.set_xlabel(f"z{j}")
                    if j == 0: a.set_ylabel(f"z{i}")
                    a.tick_params(labelsize=6)
            fig.suptitle(f"FM encoder latents Z (N={len(Z)}, {len(suites)} suites) "
                         f"— red = fitted Gaussian 1$\\sigma$  | eff dim {eff_dim:.1f}/{d}")
            fig.tight_layout()
            fig.savefig(png_out, dpi=120)
            print(f"Saved {png_out}")
        except Exception as e:
            print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
