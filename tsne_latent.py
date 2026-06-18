"""t-SNE of the extracted FM-encoder latents, coloured by simulation suite.

Reads cached/latent_stats.npz (Z, suites, ranges from extract_latents.py) and
projects the 8-dim latent to 2D with t-SNE. Each suite (IllustrisTNG / Astrid /
SIMBA) gets its own colour. If the suites separate into disjoint clusters, the
latent is encoding feedback-model identity rather than a smooth feedback axis.

    python tsne_latent.py [--npz cached/latent_stats.npz] [--perplexity 40]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="cached/latent_stats.npz")
    ap.add_argument("--out", default="cached/latent_tsne.png")
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    Z = d["Z"].astype(np.float32)
    suites = [str(s) for s in d["suites"]]
    ranges = d["ranges"]
    print(f"Z {Z.shape}  suites {suites}")

    # per-row suite label from the saved row ranges
    labels = np.empty(len(Z), dtype=int)
    for k, (lo, hi) in enumerate(ranges):
        labels[lo:hi] = k

    emb = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
               random_state=args.seed, max_iter=1000).fit_transform(Z)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, s in enumerate(suites):
        m = labels == k
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.5,
                   color=colors[k % len(colors)], label=s)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title(f"FM-encoder latent t-SNE (N={len(Z)}, dim={Z.shape[1]}, "
                 f"perp={args.perplexity:g})")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
