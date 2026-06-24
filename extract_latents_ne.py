"""Extract FM-encoder latents for an NE multi-task ckpt (encoder in_channels=2,
stacked (Mgas, ne)). Mirrors extract_latents.py but feeds the 2-ch encode input
from the Mgas_norm + Ne_norm cache. Computes latent structure (per-dim std, corr,
participation-ratio eff-dim) and an optional t-SNE coloured by suite.

  python extract_latents_ne.py --config experiments/ne/fm_ne.yaml \
      --checkpoint latent-pipeline-ne/vie2e2x0/checkpoints/best-epoch=031-val_loss=0.008031.ckpt \
      --out cached/latent_stats_ne_ep031.npz --max_per_suite 0   # 0 = all

On a GPU-less dev node use --device cpu --max_per_suite 150 for a quick subset.
"""
import os, argparse, numpy as np, yaml, torch
from module import FlowMatchingModel

CACHE = "/mnt/ceph/users/mliu1/latent_pipeline_cache/L25_LH_128_norm"
MGAS_TMPL = CACHE + "/Mgas_norm_{suite}_LH_128_z=0.0.npy"
NE_TMPL = CACHE + "/Ne_norm_{suite}_LH_128_z=0.0.npy"


def encode_suite(model, suite, dev, clamp, batch, max_n):
    mg = np.load(MGAS_TMPL.format(suite=suite), mmap_mode="r")
    ne = np.load(NE_TMPL.format(suite=suite), mmap_mode="r")
    n = len(mg) if max_n in (0, None) else min(max_n, len(mg))
    out = np.empty((n, model.gas_encoder.proj.out_features), dtype=np.float32)
    for i in range(0, n, batch):
        j = min(i + batch, n)
        a = np.asarray(mg[i:j], dtype=np.float32)
        b = np.asarray(ne[i:j], dtype=np.float32)
        x = np.stack([a, b], axis=1)                       # (B, 2, D, D, D)
        x = torch.from_numpy(x).clamp_(-clamp, clamp).to(dev)
        with torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            z = model.gas_encoder(x)
        out[i:j] = z.float().cpu().numpy()
        print(f"    {suite}: {j}/{n}", flush=True)
    return out


def eff_dim(cov):
    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 0, None)
    return (w.sum() ** 2) / (np.square(w).sum() + 1e-12), w[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/ne/fm_ne.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="cached/latent_stats_ne.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_per_suite", type=int, default=0)
    ap.add_argument("--tsne_out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    suites = cfg["data"]["suites"]
    clamp = float(cfg["data"].get("clamp_val", 10))
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    print(f"Loading {args.checkpoint} (use_ne={cfg['data'].get('use_ne')})")
    model = FlowMatchingModel.load_from_checkpoint(
        args.checkpoint, cfg=cfg, strict=False).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"latent_dim={model.gas_encoder.proj.out_features} | suites={suites} | dev={dev}")

    Z_parts, ranges, off = [], {}, 0
    for s in suites:
        z = encode_suite(model, s, dev, clamp, args.batch, args.max_per_suite)
        Z_parts.append(z); ranges[s] = (off, off + len(z)); off += len(z)
    Z = np.concatenate(Z_parts, 0).astype(np.float32)

    mean, std = Z.mean(0), Z.std(0)
    cov = np.cov(Z, rowvar=False)
    corr = np.corrcoef(Z, rowvar=False)
    ed, eig = eff_dim(cov)
    offdiag = np.abs(corr[~np.eye(len(mean), dtype=bool)])
    np.set_printoptions(precision=3, suppress=True)
    print(f"\nZ {Z.shape}  min {Z.min():.3f} max {Z.max():.3f}")
    print("per-dim std :", std)
    print("corr:\n", corr)
    print(f"max|off-diag corr| {offdiag.max():.3f}  mean {offdiag.mean():.3f}")
    print(f"participation-ratio eff-dim = {ed:.2f} / {len(mean)}")
    print("cov eigen-spectrum:", eig)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, Z=Z, mean=mean, cov=cov, std=std, corr=corr,
             eff_dim=ed, eig=eig, suites=np.array(suites),
             ranges=np.array([ranges[s] for s in suites]))
    print("saved", args.out)

    if args.tsne_out:
        from sklearn.manifold import TSNE
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        emb = TSNE(n_components=2, init="pca", perplexity=30,
                   random_state=42).fit_transform(Z)
        plt.figure(figsize=(6, 5))
        for s in suites:
            a, b = ranges[s]
            plt.scatter(emb[a:b, 0], emb[a:b, 1], s=4, alpha=0.5, label=s)
        plt.legend(); plt.title(f"ne latent t-SNE (eff-dim {ed:.2f}/{len(mean)})")
        plt.tight_layout(); plt.savefig(args.tsne_out, dpi=130)
        print("saved", args.tsne_out)


if __name__ == "__main__":
    main()
