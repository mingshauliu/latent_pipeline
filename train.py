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
                  MultiSuiteFMDataset, load_cache_pool, CachedFMDataset,
                  load_velocity_arrs, load_target_arrs)
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
    """Load a source state_dict into a model whose latent head / FiLM-fusion / channel
    counts changed shape. Copies all shape-matching tensors, then smart-inits the
    reshaped layers from source slices:
      - gas_encoder.proj <- source rows into the new mu rows (latent_dim / variational).
      - net.cond_fuse.0  <- source time(64)+cosmo(64) input cols (latent cols fresh).
      - input-channel expansion (UNet in_channels grows: +velocity, +ne x_t channels).
      - output-channel expansion (out_conv 1->2 when adding the ne target: copy the Mgas
        out row, leave the ne row at its zero-init).
    The first UNet conv (net.enc1.conv1/skip) input layout is [x_t(out_ch), nbody, vel?];
    when out_channels grows (Mgas->Mgas+ne) the static nbody/vel cols SHIFT, so they are
    remapped by role rather than naively column-copied. EMA shadow is NOT loaded."""
    own = model.state_dict()
    src_oc = int(src_sd["net.out_conv.weight"].shape[0]) if "net.out_conv.weight" in src_sd else 1
    dst_oc = int(model.out_channels)
    enc1_in = {"net.enc1.conv1.weight", "net.enc1.skip.weight"}
    copied, skipped, chan = [], [], []
    for k, v in src_sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            copied.append(k)
        elif (k in enc1_in and k in own and dst_oc != src_oc
              and own[k].shape[1] > v.shape[1] and own[k].shape[0] == v.shape[0]
              and own[k].shape[2:] == v.shape[2:]):
            # first conv: input layout [x_t(oc), nbody, vel?]. Remap by ROLE so the
            # static nbody/vel cols land in their shifted positions (extra x_t cols
            # for the new ne channel stay fresh).
            own[k][:, :src_oc].copy_(v[:, :src_oc])                 # x_t (Mgas) cols
            own[k][:, dst_oc:dst_oc + (v.shape[1] - src_oc)].copy_(v[:, src_oc:])  # nbody[+vel]
            chan.append((k + " (role-remap)", tuple(v.shape), tuple(own[k].shape)))
        elif (k in own and v.dim() == own[k].dim() and v.dim() >= 2
              and own[k].shape[0] == v.shape[0] and own[k].shape[2:] == v.shape[2:]
              and own[k].shape[1] > v.shape[1]):
            # input-channel expansion (e.g. UNet in_channels 2->3 for velocity): copy the
            # existing input cols, leave the new col(s) fresh. Encoder stem 1->2 (ne) too.
            c = v.shape[1]
            own[k][:, :c].copy_(v)
            chan.append((k, tuple(v.shape), tuple(own[k].shape)))
        elif (k in own and v.dim() == own[k].dim() and v.dim() >= 1
              and own[k].shape[0] > v.shape[0] and own[k].shape[1:] == v.shape[1:]):
            # output-channel expansion (out_conv 1->2 for the ne target): copy the Mgas
            # out row(s); the ne row keeps its zero-init (designed v=0 start).
            r = v.shape[0]
            own[k][:r].copy_(v)
            chan.append((k + " (out-expand)", tuple(v.shape), tuple(own[k].shape)))
        else:
            skipped.append((k, tuple(v.shape), tuple(own[k].shape) if k in own else None))

    pw = "gas_encoder.proj.weight"
    if pw in src_sd and pw in own and src_sd[pw].shape[1] == own[pw].shape[1]:
        # only when the encoder output width matches (encoder_base unchanged). If
        # encoder_base differs the encoder is fresh anyway -> leave proj fresh too.
        s, dnew = src_sd[pw], own[pw]
        # cap at latent_dim so only the mu head is warm-started; for a variational
        # head the logvar rows [latent_dim:] stay freshly initialised (logvar~0).
        r = min(s.shape[0], dnew.shape[0], model.latent_dim)
        dnew[:r].copy_(s[:r])
        own["gas_encoder.proj.bias"][:r].copy_(src_sd["gas_encoder.proj.bias"][:r])
        print(f"  smart-init gas_encoder.proj: copied {r} (mu) rows from source")
    elif pw in src_sd and pw in own:
        print(f"  gas_encoder.proj left fresh (encoder width "
              f"{src_sd[pw].shape[1]}->{own[pw].shape[1]}; encoder_base changed)")

    cw = "net.cond_fuse.0.weight"
    if cw in src_sd and cw in own:
        s, dnew = src_sd[cw], own[cw]
        c = min(128, s.shape[1], dnew.shape[1])   # time(64)+cosmo(64); latent cols fresh
        dnew[:, :c].copy_(s[:, :c])
        own["net.cond_fuse.0.bias"].copy_(src_sd["net.cond_fuse.0.bias"])
        print(f"  smart-init net.cond_fuse.0: copied {c} time+cosmo cols from source")

    model.load_state_dict(own, strict=True)
    print(f"Partial warm-start: copied {len(copied)} matching tensors; "
          f"{len(chan)} input-channel-expanded; {len(skipped)} reshaped (smart-init/fresh):")
    for k, sshape, dshape in chan:
        print(f"  in-chan-expand {k}: src{sshape} -> dst{dshape} (extra col fresh)")
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
    use_velocity = d.get("use_velocity", False)
    # ordered extra-OUTPUT targets; back-compat: use_ne -> [ne]. e.g. [ne, T].
    target_fields = d.get("target_fields")
    if target_fields is None:
        target_fields = ["ne"] if d.get("use_ne", False) else []
    if use_cache:
        print("Loading pre-normalised cache pool:")
        nbody_arrs, mgas_arrs, cosmo_all, flat = load_cache_pool(d)
        clamp_val = d.get("clamp_val", 10.0)
        vel_arrs = load_velocity_arrs(d) if use_velocity else None
        target_arrs = [load_target_arrs(d, f) for f in target_fields]
        if use_velocity:
            print(f"  + velocity channel ({len(vel_arrs)} suites) -> in_channels +1")
        if target_fields:
            print(f"  + extra TARGET channels {target_fields} -> "
                  f"out_channels={1+len(target_fields)}, in_channels=out_channels+1[+vel]")
        ds_cls = lambda ix, aug: CachedFMDataset(  # noqa: E731
            nbody_arrs, mgas_arrs, cosmo_all, flat, ix,
            crop_size=crop, augment=aug, clamp_val=clamp_val,
            vel_arrs=vel_arrs, target_arrs=target_arrs)
    else:
        if use_velocity or target_fields:
            raise ValueError("use_velocity/target_fields require use_cache=true (cache paths)")
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
