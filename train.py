"""Train the FM branch: multi-suite Nbody -> Mgas with an 8-dim Mgas latent.

    python train.py --config config/config.yaml
"""

import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml

from data import (load_suite_pool, compute_field_stats, compute_cosmo_stats,
                  MultiSuiteFMDataset, load_cache_pool, CachedFMDataset)
from module import FlowMatchingModel

NORM_PATH = "cached/norm_latent.npz"


def build_norm(d, nbody_arrs, mgas_arrs, cosmo_all, flat):
    """Compute (or load) log1p+z-score stats over the pool. Cached to NORM_PATH."""
    if os.path.exists(NORM_PATH):
        z = np.load(NORM_PATH)
        print(f"Loaded norm stats from {NORM_PATH}")
        return {k: z[k] for k in z.files}
    n_stat = d.get("stats_n_sample", 64)
    nb_mean, nb_std = compute_field_stats(nbody_arrs, flat, n_stat, seed=0)
    mg_mean, mg_std = compute_field_stats(mgas_arrs, flat, n_stat, seed=1)
    cm_mean, cm_std = compute_cosmo_stats(cosmo_all)
    norm = dict(nbody_mean=np.float32(nb_mean), nbody_std=np.float32(nb_std),
                mgas_mean=np.float32(mg_mean), mgas_std=np.float32(mg_std),
                cosmo_mean=cm_mean, cosmo_std=cm_std)
    os.makedirs(os.path.dirname(NORM_PATH), exist_ok=True)
    np.savez(NORM_PATH, **norm)
    print(f"Computed + saved norm stats -> {NORM_PATH}")
    print(f"  nbody log1p mean/std = {nb_mean:.4f}/{nb_std:.4f}")
    print(f"  mgas  log1p mean/std = {mg_mean:.4f}/{mg_std:.4f}")
    print(f"  cosmo mean/std = {cm_mean} / {cm_std}")
    return norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    d, t = cfg["data"], cfg["training"]
    seed = t.get("seed")
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    crop = d.get("crop_size")
    if crop and crop >= d["resolution"]:
        crop = None

    use_cache = d.get("use_cache", False)
    if use_cache:
        print("Loading pre-normalised cache pool:")
        nbody_arrs, mgas_arrs, cosmo_all, flat = load_cache_pool(d)
        clamp_val = d.get("clamp_val", 10.0)
        ds_cls = lambda ix, aug: CachedFMDataset(  # noqa: E731
            nbody_arrs, mgas_arrs, cosmo_all, flat, ix,
            crop_size=crop, augment=aug, clamp_val=clamp_val)
    else:
        print("Loading multi-suite pool (lazy norm):")
        nbody_arrs, mgas_arrs, cosmo_all, flat = load_suite_pool(d)
        norm = build_norm(d, nbody_arrs, mgas_arrs, cosmo_all, flat)
        ds_cls = lambda ix, aug: MultiSuiteFMDataset(  # noqa: E731
            nbody_arrs, mgas_arrs, cosmo_all, flat, ix, norm,
            crop_size=crop, augment=aug)

    n = len(flat)
    n_val = int(n * d["val_split"])
    rng = np.random.RandomState(seed or 42)
    idx = rng.permutation(n)
    tr_idx, va_idx = idx[:n - n_val], idx[n - n_val:]

    tr_ds = ds_cls(tr_idx, True)
    va_ds = ds_cls(va_idx, False)
    pw = t["num_workers"] > 0
    pf = t.get("prefetch_factor", 2) if pw else None
    kw = dict(pin_memory=True, persistent_workers=pw, prefetch_factor=pf, drop_last=True)
    tr_dl = DataLoader(tr_ds, batch_size=t["batch_size"], shuffle=True,
                       num_workers=t["num_workers"], **kw)
    kw["drop_last"] = False
    va_dl = DataLoader(va_ds, batch_size=t["batch_size"], shuffle=False,
                       num_workers=t["num_workers"], **kw)

    model = FlowMatchingModel(cfg)

    trainer = pl.Trainer(
        logger=WandbLogger(project=t.get("wandb_project", "latent-pipeline"), log_model=False),
        max_epochs=t["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=t["devices"] if not args.fast_dev_run else 1,
        strategy=t["strategy"] if not args.fast_dev_run else "auto",
        precision=t["precision"],
        gradient_clip_val=t["gradient_clip"],
        accumulate_grad_batches=t["accumulate_grad"],
        log_every_n_steps=t["log_every_n_steps"],
        check_val_every_n_epoch=1,
        fast_dev_run=args.fast_dev_run,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", filename="best-{epoch:03d}-{val_loss:.6f}",
                            save_top_k=t.get("save_top_k", 3), mode="min", save_last=True),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        num_sanity_val_steps=2,
    )
    trainer.fit(model, tr_dl, va_dl, ckpt_path=t.get("resume_from"))
    print(f"Best: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
