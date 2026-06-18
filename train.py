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


@torch.no_grad()
def warm_load_partial(model, src_sd):
    """Load a source state_dict into a model whose latent head / FiLM-fusion layers
    changed shape (latent_dim and/or variational mu,logvar head differ). Copies all
    shape-matching tensors, then smart-inits the two reshaped layers from source
    slices: gas_encoder.proj <- source rows into the new mu rows; net.cond_fuse.0 <-
    source time(64)+cosmo(64) input columns (latent columns stay freshly initialised).
    EMA shadow is intentionally NOT loaded — it rebuilds from current weights."""
    own = model.state_dict()
    copied, skipped = [], []
    for k, v in src_sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            copied.append(k)
        else:
            skipped.append((k, tuple(v.shape), tuple(own[k].shape) if k in own else None))

    pw = "gas_encoder.proj.weight"
    if pw in src_sd and pw in own:
        s, dnew = src_sd[pw], own[pw]
        # cap at latent_dim so only the mu head is warm-started; for a variational
        # head the logvar rows [latent_dim:] stay freshly initialised (logvar~0).
        r = min(s.shape[0], dnew.shape[0], model.latent_dim)
        dnew[:r].copy_(s[:r])
        own["gas_encoder.proj.bias"][:r].copy_(src_sd["gas_encoder.proj.bias"][:r])
        print(f"  smart-init gas_encoder.proj: copied {r} (mu) rows from source")

    cw = "net.cond_fuse.0.weight"
    if cw in src_sd and cw in own:
        s, dnew = src_sd[cw], own[cw]
        c = min(128, s.shape[1], dnew.shape[1])   # time(64)+cosmo(64); latent cols fresh
        dnew[:, :c].copy_(s[:, :c])
        own["net.cond_fuse.0.bias"].copy_(src_sd["net.cond_fuse.0.bias"])
        print(f"  smart-init net.cond_fuse.0: copied {c} time+cosmo cols from source")

    model.load_state_dict(own, strict=True)
    print(f"Partial warm-start: copied {len(copied)} matching tensors; "
          f"{len(skipped)} reshaped (smart-init/fresh):")
    for k, sshape, dshape in skipped:
        print(f"  reshaped {k}: src{sshape} -> dst{dshape}")


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

    # Weights-only warm start: load the converged EMA-baked weights but start a
    # FRESH optimizer/scheduler at epoch 0 (unlike resume_from, which restores the
    # old scheduler state + epoch). Use this to test a new/aggressive lr schedule
    # against a plateaued checkpoint. EMA shadow is carried over so EMA keeps refining.
    init_from = t.get("init_from")
    if init_from and not t.get("resume_from"):
        print(f"Warm-start (weights only) from: {init_from}")
        ck = torch.load(init_from, map_location="cpu", weights_only=False)
        if t.get("init_strict", True):
            model.load_state_dict(ck["state_dict"], strict=True)  # EMA-baked weights
            if model.ema_enabled and "ema_shadow" in ck:
                model._ema_shadow = {n: tt for n, tt in ck["ema_shadow"].items()}
        else:
            # Partial warm-start: the latent head (gas_encoder.proj) and the FiLM
            # context fusion (net.cond_fuse.0) change shape when latent_dim and/or
            # the variational (mu,logvar) head differ from the source ckpt. Copy
            # every shape-matching key (full UNet backbone + encoder SE-ResNet), then
            # smart-init the two reshaped layers from their source slices. EMA shadow
            # is NOT carried over (shapes changed) — it rebuilds from current weights.
            warm_load_partial(model, ck["state_dict"])

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
