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
from infer import load_norm, norm_field, norm_nbody, nbody_div, norm_vel, TARGET_SPEC

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
    ap.add_argument("--noise_std", type=float, default=None,
                    help="override sampling noise to match the ckpt's training value "
                         "(sv6mp9wt=0.1; config default is now 0.2 for new runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.noise_std is not None:
        cfg["training"]["noise_std"] = args.noise_std
    dev = args.device if torch.cuda.is_available() else "cpu"
    norm = load_norm()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.checkpoint}")
    model = FlowMatchingModel.load_from_checkpoint(
        args.checkpoint, cfg=cfg, strict=False).to(dev).eval()
    ldim = cfg["model"].get("latent_dim", 8)
    box = cfg["data"]["box_size"]
    print(f"latent_dim={ldim} | box={box} | mode={args.latent_mode} | dev={dev}")

    # multi-task / velocity ckpts (in5/out3+vel etc.) — mirror infer.py's wiring:
    # vel = extra CONDITIONING channel, extras (ne/T) = extra OUTPUT channels; the
    # encoder encodes stacked (Mgas, *extras). Metrics stay on the Mgas channel (0).
    use_velocity = cfg["data"].get("use_velocity", False)
    target_fields = cfg["data"].get("target_fields")
    if target_fields is None:
        target_fields = ["ne"] if cfg["data"].get("use_ne", False) else []
    if use_velocity:
        assert "nbodyvel_mean" in norm, "use_velocity needs nbodyvel stats in norm_latent.npz"

    srcs = cfg["inference"]["sources"]
    src = next((s for s in srcs if args.suite and args.suite in s["name"]), srcs[0])
    suite = src["name"]
    print(f"source: {suite} | use_velocity={use_velocity} | target_fields={target_fields}")
    nbody = np.load(src["nbody_path"], mmap_mode="r")
    mgas = np.load(src["mgas_path"], mmap_mode="r")
    vel_arr = np.load(src["vel_path"], mmap_mode="r") if use_velocity else None
    target_refs = []
    if args.latent_mode == "encode":
        for f in target_fields:
            pk = TARGET_SPEC[f]["path"]
            assert src.get(pk), f"encode + target '{f}' needs src.{pk}"
            target_refs.append(np.load(src[pk], mmap_mode="r"))
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
        vel_t = None
        if use_velocity:
            vv = norm_vel(np.asarray(vel_arr[i], dtype=np.float32),
                          norm["nbodyvel_mean"], norm["nbodyvel_std"])
            if clamp_val is not None:
                np.clip(vv, -clamp_val, clamp_val, out=vv)
            vel_t = torch.from_numpy(vv)[None, None].to(dev)
        cosmo = (cosmo_all[i] - norm["cosmo_mean"]) / norm["cosmo_std"]
        cosmo_t = torch.from_numpy(cosmo.astype(np.float32))[None].to(dev)

        if args.latent_mode == "encode":
            enc_in = torch.from_numpy(true_n)[None, None].to(dev)
            for f, ref in zip(target_fields, target_refs):
                spec = TARGET_SPEC[f]
                tv_ = spec["norm"](np.asarray(ref[i], dtype=np.float32),
                                   norm[spec["mkey"]], norm[spec["skey"]])
                if clamp_val is not None:
                    np.clip(tv_, -clamp_val, clamp_val, out=tv_)
                enc_in = torch.cat([enc_in, torch.from_numpy(tv_)[None, None].to(dev)], dim=1)
            with torch.no_grad():
                enc = model.gas_encoder(enc_in)
            latent = enc[0] if isinstance(enc, tuple) else enc
        else:
            latent = torch.zeros(1, ldim, device=dev)

        with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            s = model.sample(nb_t, cosmo_t, latent, num_steps=args.num_steps,
                             method=args.method, vel=vel_t)
        synth_n = s[0, 0].float().cpu().numpy()   # Mgas channel, normed log space
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
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')
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
    
    if HAS_PKL:
        from matplotlib.gridspec import GridSpec
        fig2 = plt.figure(figsize=(7, 9))
        gs = GridSpec(3, 1, height_ratios=[3, 3, 1], hspace=0.15)
        ax_hist = fig2.add_subplot(gs[0])
        ax_pk = fig2.add_subplot(gs[1])
        ax_ratio = fig2.add_subplot(gs[2], sharex=ax_pk)
        ax2 = [ax_hist, ax_pk, ax_ratio]
    else:
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4.2))
        ax2 = np.atleast_1d(ax2)
    
    # ── (b1) 2D histogram ──────────────────────────────────────────────────────
    lo, hi = min(tv.min(), sv.min()), max(tv.max(), sv.max())
    fig_hist, ax_hist_only = plt.subplots(figsize=(5, 4.2))
    h = ax_hist_only.hist2d(tv, sv, bins=200, range=[[lo, hi], [lo, hi]], cmap="inferno",
                            norm=matplotlib.colors.LogNorm())
    ax_hist_only.plot([lo, hi], [lo, hi], "w--", lw=1)
    plt.colorbar(h[3], ax=ax_hist_only, fraction=0.046)
    ax_hist_only.set_xlabel("true (norm log Mgas)"); ax_hist_only.set_ylabel("synth (norm log Mgas)")
    ax_hist_only.set_title(f"voxel truth-vs-pred ({N} cubes)")
    fig_hist.tight_layout()
    p_hist = os.path.join(args.out_dir, f"indist_{suite}_hist.png")
    fig_hist.savefig(p_hist, dpi=130); print(f"Saved {p_hist}")

    # ── (b2) P(k) + ratio stacked ──────────────────────────────────────────────
    if HAS_PKL:
        ks, pts, pss, rats, xcs = None, [], [], [], []
        for i in range(N):
            k, pt = pk_1d(true_raw[i], box)     # demeaned normed-log field
            _, ps = pk_1d(synth_raw[i], box)
            ks = k; pts.append(pt); pss.append(ps); rats.append(ps / (pt + 1e-30))
            xcs.append(xcorr_metric(synth_raw[i], true_raw[i], box))
        pt_m, ps_m, r_m = np.mean(pts, 0), np.mean(pss, 0), np.mean(rats, 0)
        
        fig2 = plt.figure(figsize=(7, 6.5))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
        ax_pk = fig2.add_subplot(gs[0])
        ax_ratio = fig2.add_subplot(gs[1], sharex=ax_pk)
        
        ax_pk.loglog(ks, pt_m, "C0-", label="true", color="tab:blue")
        ax_pk.loglog(ks, ps_m, "C0--", label="synth", color="tab:blue")
        ax_pk.axvline(15, color="gray", ls=":", lw=1)
        ax_pk.set_xlim(ks.min(), 15)
        ax_pk.set_ylabel("P(k) of norm-log Mgas")
        ax_pk.legend(); ax_pk.set_title("mean power spectrum (log space)")
        
        ax_ratio.semilogx(ks, r_m, color="tab:blue", lw=2)
        ax_ratio.axhline(1, color="k", lw=0.8)
        ax_ratio.axhspan(0.9, 1.1, color="g", alpha=0.12)
        ax_ratio.axvline(15, color="gray", ls=":", lw=1)
        ax_ratio.set_xlim(ks.min(), 15)
        ax_ratio.set_ylim(0.5, 1.5); ax_ratio.set_xlabel("k [h/Mpc]")
        ax_ratio.set_ylabel("P_synth / P_true"); ax_ratio.set_title("P(k) ratio (±10% band)")
        
        fig2.suptitle(f"{suite} det3 in-dist (norm log) — mean xcorr={np.mean(xcs):.4f}", fontsize=11)
        fig2.tight_layout()
        p2 = os.path.join(args.out_dir, f"indist_{suite}_pk.png")
        fig2.savefig(p2, dpi=130); print(f"Saved {p2}")
        print(f"\nmean xcorr={np.mean(xcs):.4f} | P(k) ratio @k<15 mean={np.mean(r_m[ks<=15]):.4f}")


if __name__ == "__main__":
    main()
