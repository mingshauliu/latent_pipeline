"""NF truth-vs-pred on FM-SYNTH Mgas — the synth-fidelity judge.

Generates FM-synth Mgas (dim8 FM) from the Nbody of the NF held-out VAL split
(same split/seed as NF training) and feeds each synth cube through the trained
REAL-NF (nf_ck_real) -> posterior (Omega_m, sigma_8). If the FM synth retains the
per-cube cosmology signal, real-NF recovers truth ~ as well as on real Mgas
(plot_nf_2param --data_mode real: Om R2=0.966, s8 R2=0.886).

This is the FAST fidelity probe (no synth-NF retrain): real-NF never saw synth, so
train/val of the FM/NF pools is irrelevant to it; we still use the NF val indices so
the truth-vs-pred is directly comparable to the real baseline.

latent_mode:
  mean   : latent = 0 (deploy-realistic — latent unknown at sampling). THE pipeline metric.
  encode : latent encoded from the TRUE Mgas (oracle upper bound; same as eval_indist).

    python eval_nf_synth.py --fm_checkpoint <fm.ckpt> \
        --nf_checkpoint nf_ck_real/nf-752--5.4560.ckpt \
        --latent_mode mean --max_cubes 300
GPU (FM dopri5 128^3 + NF 2000-sample posterior). Writes outputs/nf_synth_<mode>_tvp.png.
"""
import argparse, os
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from module import FlowMatchingModel
from infer import load_norm, norm_field, norm_nbody, nbody_div, denorm_field
from data import load_nf_pool, nf_mgas_stats, read_cube
from nf.predict import load_nf, predict_with_uncertainty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--fm_checkpoint", required=True)
    ap.add_argument("--nf_checkpoint", required=True)
    ap.add_argument("--latent_mode", default="mean", choices=["mean", "encode"])
    ap.add_argument("--max_cubes", type=int, default=300, help="subsample of the NF val split")
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--method", default="dopri5")
    ap.add_argument("--n_posterior", type=int, default=2000)
    ap.add_argument("--noise_std", type=float, default=None,
                    help="override sampling noise to MATCH how the ckpt was trained "
                         "(sv6mp9wt was 0.1; config default is now 0.2 for new runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.noise_std is not None:
        cfg["training"]["noise_std"] = args.noise_std
    nf_data = cfg["nf"]["data"]; nf_train = cfg["nf"]["training"]
    nf_data["mode"] = "real"   # we use the real pool ONLY for indices + truth cosmo
    dev = args.device if torch.cuda.is_available() else "cpu"
    norm = load_norm()
    ldim = cfg["model"].get("latent_dim", 8)
    box = cfg["data"]["box_size"]
    clamp_val = cfg["data"].get("clamp_val", 10.0)
    os.makedirs("outputs", exist_ok=True)

    # ── NF val split (same seed/split as training) ───────────────────────────────
    mgas_src, cosmo_all, flat = load_nf_pool(nf_data)
    nf_mean, nf_std = nf_mgas_stats(nf_data)
    suites = nf_data["suites"]
    n = len(flat)
    rng = np.random.RandomState(nf_train.get("seed", 42))
    idx = rng.permutation(n)
    n_val = max(1, int(n * nf_data.get("val_split", 0.2)))
    va_idx = idx[:n_val]
    if len(va_idx) > args.max_cubes:
        va_idx = np.random.RandomState(0).choice(va_idx, args.max_cubes, replace=False)
    va_idx = np.sort(va_idx)
    print(f"NF val pool {n_val}; using {len(va_idx)} cubes | latent_mode={args.latent_mode}")

    # ── Nbody / param mmaps per suite (FM source) ────────────────────────────────
    nbody_src, param_src = {}, {}
    for suite in suites:
        nbody_src[suite] = np.load(
            cfg["data"]["nbody_path_tmpl"].format(suite=suite), mmap_mode="r")
        param_src[suite] = np.loadtxt(
            cfg["data"]["param_path_tmpl"].format(suite=suite)).astype(np.float32)

    print(f"Loading FM {args.fm_checkpoint}")
    fm = FlowMatchingModel.load_from_checkpoint(
        args.fm_checkpoint, cfg=cfg, strict=False).to(dev).eval()

    # ── generate synth, normalise EXACTLY as the NF dataset does ─────────────────
    nf_inputs, truth = [], []
    for c, g in enumerate(va_idx):
        si, li = flat[g]
        suite = suites[si]
        div = nbody_div(suite, cfg)
        nb = norm_nbody(np.asarray(nbody_src[suite][li], dtype=np.float32),
                        norm["nbody_mean"], norm["nbody_std"], div=div)
        np.clip(nb, -clamp_val, clamp_val, out=nb)
        nb_t = torch.from_numpy(nb)[None, None].to(dev)
        cosmo = (param_src[suite][li, :2] - norm["cosmo_mean"]) / norm["cosmo_std"]
        cosmo_t = torch.from_numpy(cosmo.astype(np.float32))[None].to(dev)

        if args.latent_mode == "encode":
            true_n = norm_field(read_cube(mgas_src[si][li]).astype(np.float32),
                                norm["mgas_mean"], norm["mgas_std"])
            np.clip(true_n, -clamp_val, clamp_val, out=true_n)
            with torch.no_grad():
                enc = fm.gas_encoder(torch.from_numpy(true_n)[None, None].to(dev))
            latent = enc[0] if isinstance(enc, tuple) else enc
        else:
            latent = torch.zeros(1, ldim, device=dev)

        with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            s = fm.sample(nb_t, cosmo_t, latent, num_steps=args.num_steps, method=args.method)
        synth_n = s[0, 0].float().cpu().numpy()
        # FM-native normed-log -> physical (expm1) -> NF norm3d-gas re-norm (the exact
        # deploy path; FM and NF share norm3d gas stats so this is ~identity but we do
        # it honestly in case the two stats ever diverge).
        phys = denorm_field(synth_n, norm["mgas_mean"], norm["mgas_std"])
        nf_in = (np.log1p(phys) - nf_mean) / nf_std
        nf_inputs.append(torch.from_numpy(nf_in.astype(np.float32))[None])
        truth.append(param_src[suite][li, :2].copy())
        if (c + 1) % 50 == 0:
            print(f"  generated {c + 1}/{len(va_idx)}")

    # ── NF posterior over the synth cubes ────────────────────────────────────────
    X = torch.stack(nf_inputs)                       # (N,1,128,128,128)
    Y = torch.from_numpy(np.asarray(truth, dtype=np.float32))
    loader = DataLoader(TensorDataset(X, Y), batch_size=2, shuffle=False)
    print(f"Loading NF {args.nf_checkpoint}; running {args.n_posterior}-sample posterior")
    nf = load_nf(args.nf_checkpoint, device=dev)
    y_true, y_mean, y_std, _ = predict_with_uncertainty(nf, loader, args.n_posterior, dev)

    # ── truth-vs-pred (mirrors plot_nf_2param) ───────────────────────────────────
    names = [r"$\Omega_m$", r"$\sigma_8$"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rmse_per, r2_per = [], []
    for j, (ax, label) in enumerate(zip(axes, names)):
        x, yp, err = y_true[:, j], y_mean[:, j], y_std[:, j]
        rmse = float(np.sqrt(np.mean((x - yp) ** 2)))
        r2 = 1.0 - np.sum((x - yp) ** 2) / (np.sum((x - x.mean()) ** 2) + 1e-12)
        rmse_per.append(rmse); r2_per.append(r2)
        xmin, xmax = float(x.min()), float(x.max())
        m = 0.05 * (xmax - xmin) if xmax > xmin else 0.1
        line = np.linspace(xmin - m, xmax + m, 100)
        ax.errorbar(x, yp, yerr=err, fmt="o", ms=4, alpha=0.5, color="tab:orange",
                    elinewidth=0.5, capsize=1)
        ax.plot(line, line, "r--", lw=2)
        ax.set_xlabel("Truth"); ax.set_ylabel("Prediction")
        ax.set_title(f"{label}  RMSE={rmse:.4f}  R²={r2:.3f}")
        ax.set_xlim(xmin - m, xmax + m); ax.set_ylim(xmin - m, xmax + m)
        ax.grid(alpha=0.3)
    stat = "  ".join(f"{n}: RMSE={r:.4f}, R²={s:.3f}"
                     for n, r, s in zip(names, rmse_per, r2_per))
    plt.suptitle(f"real-NF on FM-synth Mgas (dim8, latent={args.latent_mode}, N={len(y_true)})\n{stat}",
                 fontsize=13)
    plt.tight_layout()
    out = args.out or f"outputs/nf_synth_dim8_{args.latent_mode}_tvp.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}\n{stat}")


if __name__ == "__main__":
    main()
