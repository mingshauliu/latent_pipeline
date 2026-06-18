"""FM inference: sample Mgas from Nbody for each source.

Latent modes (Mgas is unknown at sampling, so the conditioning latent must be
supplied):
  mean    : latent = 0  (marginal over feedback realisations) [default]
  sample  : latent ~ N(0, I) drawn per cube. For a VARIATIONAL (KL) checkpoint
            (model.variational=true) this is the CORRECT realistic mode — the KL
            prior is N(0,I) so the latent distribution matches by construction.
            For a deterministic (tanh) checkpoint it is mismatched (use gaussian).
  encode  : latent encoded from a reference true-Mgas cube (oracle / sanity check;
            variational -> uses the posterior mean mu).
  gaussian: latent ~ N(mu, Sigma) fit to the encoded TRAINING set
            (cached/latent_stats.npz from extract_latents.py); drawn per cube,
            clamped to [-1, 1] to respect the tanh support. The realistic-coverage
            mode for a DETERMINISTIC (tanh) checkpoint; variational uses sample.

    python infer.py --config config/config.yaml
"""

import argparse
import os
import numpy as np
import torch
import yaml

from module import FlowMatchingModel

NORM_PATH = "cached/norm_latent.npz"


def load_norm():
    z = np.load(NORM_PATH)
    return {k: z[k] for k in z.files}


def norm_field(v, mean, std):
    return (np.log1p(v) - mean) / std


def norm_nbody(v, mean, std):
    """Overdensity log1p(rho/rho_bar_cube) + global z-score — MUST match prep_cache
    (data is trained on the cached overdensity Nbody, suite-invariant)."""
    v = v.astype(np.float32)
    return (np.log1p(v / (v.mean() + 1e-8)) - mean) / std


def denorm_field(x, mean, std):
    return np.expm1(x * std + mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ic = cfg["inference"]
    norm = load_norm()
    dev = args.device if torch.cuda.is_available() else "cpu"

    ckpt = args.checkpoint or ic.get("checkpoint")
    assert ckpt, "no checkpoint given (inference.checkpoint or --checkpoint)"
    print(f"Loading {ckpt}")
    model = FlowMatchingModel.load_from_checkpoint(ckpt, cfg=cfg, strict=False).to(dev).eval()
    latent_dim = cfg["model"].get("latent_dim", 8)

    out_root = ic["output_dir"]
    n_steps = ic.get("num_steps", 100)
    method = ic.get("method", "euler")
    offload = ic.get("offload_skips", False)
    mode = ic.get("latent_mode", "mean")
    n_stoch = ic.get("n_stochastic", 1)

    gauss = None
    if mode == "gaussian":
        stats_path = ic.get("latent_stats", "cached/latent_stats.npz")
        z = np.load(stats_path)
        gmu = z["mean"].astype(np.float64)
        gcov = z["cov"].astype(np.float64)
        # eigen-decomposition sampler: robust to the (degenerate) low-rank cov —
        # L = mu + V·sqrt(max(lam,0))·randn keeps draws on the SAME latent manifold
        # as training (degenerate directions stay collapsed; no PSD warnings).
        lam, V = np.linalg.eigh(gcov)
        A = V @ np.diag(np.sqrt(np.clip(lam, 0.0, None)))
        gauss = (gmu, A)
        rank = int((lam > 1e-8 * lam.max()).sum())
        print(f"gaussian latent envelope from {stats_path} (dim={len(gmu)}, "
              f"rank={rank}/{len(gmu)} [degenerate], |mu|max={np.abs(gmu).max():.3f})")

    for src in ic.get("sources", []):
        name = src["name"]
        nbody = np.load(src["nbody_path"], mmap_mode="r")
        cosmo_all = np.loadtxt(src["param_path"]).astype(np.float32)[:, :cfg["data"].get("n_cosmo", 2)]
        mgas_ref = np.load(src["mgas_path"], mmap_mode="r") if (mode == "encode" and src.get("mgas_path")) else None
        n_take = src.get("n_samples") or len(nbody)
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{name}] {n_take} cubes, latent_mode={mode}, n_stochastic={n_stoch}")

        for i in range(n_take):
            for k in range(n_stoch):
                tag = f"_{k}" if n_stoch > 1 else ""
                out_path = os.path.join(out_dir, f"sample_{i:04d}{tag}.npy")
                if os.path.exists(out_path):
                    continue
                nb = norm_nbody(np.asarray(nbody[i], dtype=np.float32),
                                norm["nbody_mean"], norm["nbody_std"])
                nb_t = torch.from_numpy(nb)[None, None].to(dev)
                cosmo = (cosmo_all[i] - norm["cosmo_mean"]) / norm["cosmo_std"]
                cosmo_t = torch.from_numpy(cosmo.astype(np.float32))[None].to(dev)

                if mode == "mean":
                    latent = torch.zeros(1, latent_dim, device=dev)
                elif mode == "sample":
                    latent = torch.randn(1, latent_dim, device=dev)
                elif mode == "gaussian":
                    gmu, A = gauss
                    l = gmu + A @ np.random.randn(len(gmu))
                    l = np.clip(l, -1.0, 1.0).astype(np.float32)
                    latent = torch.from_numpy(l)[None].to(dev)
                elif mode == "encode":
                    mg = norm_field(np.asarray(mgas_ref[i], dtype=np.float32),
                                    norm["mgas_mean"], norm["mgas_std"])
                    with torch.no_grad():
                        enc = model.gas_encoder(torch.from_numpy(mg)[None, None].to(dev))
                    # variational encoder returns (mu, logvar) -> use the mean mu.
                    latent = enc[0] if isinstance(enc, tuple) else enc
                else:
                    raise ValueError(f"unknown latent_mode {mode}")

                with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    synth = model.sample(nb_t, cosmo_t, latent, num_steps=n_steps,
                                         method=method, offload_skips=offload)
                out = denorm_field(synth[0, 0].float().cpu().numpy(),
                                   norm["mgas_mean"], norm["mgas_std"])
                np.save(out_path, out.astype(np.float32))
        print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
