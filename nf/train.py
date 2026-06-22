"""Train multi-suite NF (Mgas -> Omega_m, sigma_8).

    python -m nf.train --config config/config.yaml [sweep overrides]
"""

import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import yaml

from data import load_nf_pool, nf_mgas_stats, load_cache_pool, velocity_stats
from .module import (LitNFRegressor, MultiSuiteNFDataModule,
                     MultiSuiteVelocityDataModule)

# 128^3 paradigm: NF reuses the SHARED norm3d 'gas' stats (see data.nf_mgas_stats)
# so FM cache, Magneticum held-out cache, and NF input normalise identically.


class WarmupGatedEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores epochs before `start_epoch` (flow warm-up)."""

    def __init__(self, start_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def _run_early_stopping_check(self, trainer):
        if trainer.current_epoch < self.start_epoch:
            return
        super()._run_early_stopping_check(trainer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--data_mode", choices=["real", "synth", "velocity"], default=None,
                   help="NF training data: real CAMELS Mgas, FM-synth Mgas, or "
                        "velocity = Mgas_norm-Nbody_norm from the FM cache (overrides config)")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--summary_dim", type=int, default=None)
    p.add_argument("--base_channels", type=int, default=None)
    p.add_argument("--flow_transforms", type=int, default=None)
    p.add_argument("--flow_hidden", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--lr_encoder", type=float, default=None)
    p.add_argument("--lr_flow", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true",
                   help="1 train+val batch, 1 device, no logging/ckpt (smoke)")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    nf_cfg = cfg["nf"]
    nf_data, nf_model, nf_train = nf_cfg["data"], nf_cfg["model"], nf_cfg["training"]
    if args.data_mode:
        nf_data["mode"] = args.data_mode
    if args.checkpoint_dir:
        nf_train["checkpoint_dir"] = args.checkpoint_dir
    for k in ("summary_dim", "base_channels", "flow_transforms", "flow_hidden", "dropout"):
        if getattr(args, k) is not None:
            nf_model[k] = getattr(args, k)
    for k in ("weight_decay", "lr_encoder", "lr_flow", "warmup_epochs"):
        if getattr(args, k) is not None:
            nf_train[k] = getattr(args, k)

    pl.seed_everything(nf_train.get("seed", 42), workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    mode = nf_data.get("mode", "real")
    print(f"NF data mode = {mode}")
    if mode == "velocity":
        # velocity-NF: v = Mgas_norm - Nbody_norm from the FM cache (cfg['data']).
        clamp_val = cfg["data"].get("clamp_val", 10.0)
        nbody_arrs, mgas_arrs, cosmo_all, flat = load_cache_pool(cfg["data"])
        vel_mean, vel_std = velocity_stats(
            cfg["data"], cache_path=nf_data.get("velocity_stats", "cached/norm_velocity.npz"),
            clamp_val=clamp_val)
        print(f"velocity norm: mean/std {vel_mean:.4f}/{vel_std:.4f} (clamp {clamp_val})")
        dm = MultiSuiteVelocityDataModule(
            nbody_arrs, mgas_arrs, cosmo_all, flat, vel_mean, vel_std,
            clamp_val=clamp_val, val_split=nf_cfg["data"].get("val_split", 0.2),
            batch_size=nf_train["batch_size"], num_workers=nf_train.get("num_workers", 4),
            seed=nf_train.get("seed", 42), input_noise_std=nf_train.get("input_noise_std", 0.0))
    else:
        mgas_src, cosmo_all, flat = load_nf_pool(nf_data)
        mgas_mean, mgas_std = nf_mgas_stats(nf_data)
        print(f"NF Mgas norm (shared norm3d gas): mean/std {mgas_mean:.4f}/{mgas_std:.4f}")
        dm = MultiSuiteNFDataModule(
            mgas_src, cosmo_all, flat, mgas_mean, mgas_std,
            val_split=nf_cfg["data"].get("val_split", 0.2),
            batch_size=nf_train["batch_size"], num_workers=nf_train.get("num_workers", 4),
            seed=nf_train.get("seed", 42), input_noise_std=nf_train.get("input_noise_std", 0.0))
    dm.setup()

    model = LitNFRegressor(
        lr_encoder=nf_train["lr_encoder"], lr_flow=nf_train["lr_flow"],
        weight_decay=nf_train.get("weight_decay", 1e-4),
        base_channels=nf_model["base_channels"], num_params=nf_model["num_params"],
        flow_hidden=nf_model.get("flow_hidden", 128),
        flow_transforms=nf_model.get("flow_transforms", 4),
        flow_type=nf_model.get("flow_type", "nsf"), flow_bins=nf_model.get("flow_bins", 8),
        dropout=nf_model.get("dropout", 0.15), summary_dim=nf_model.get("summary_dim"),
        circular_padding=nf_model.get("circular_padding", True),
        warmup_epochs=nf_train.get("warmup_epochs", 20), max_epochs=nf_train["max_epochs"],
        flow_warmup_epochs=nf_train.get("flow_warmup_epochs", 50),
        aux_loss_weight=nf_train.get("aux_loss_weight", 0.5),
        aux_loss_decay=nf_train.get("aux_loss_decay", 1.0),
        skip_nan_loss=nf_train.get("skip_nan_loss", True),
        target_mean=dm.target_mean, target_std=dm.target_std,
        plot_every_n_epochs=nf_train.get("plot_every_n_epochs", 5),
        param_names=nf_cfg.get("inference", {}).get("param_names"))

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    ckpt_dir = nf_train.get("checkpoint_dir", "nf_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, save_top_k=3, monitor="val/nll", mode="min",
                        filename="nf-{epoch:03d}-{val/nll:.4f}", save_last=True,
                        auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    es = nf_train.get("early_stop_patience")
    if es:
        callbacks.append(WarmupGatedEarlyStopping(
            monitor="val/nll", mode="min", patience=int(es),
            start_epoch=int(nf_train.get("flow_warmup_epochs", 0)), verbose=True))

    trainer = pl.Trainer(
        fast_dev_run=args.fast_dev_run,
        max_epochs=nf_train["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if args.fast_dev_run else nf_train.get("devices", 4),
        strategy="auto" if args.fast_dev_run else nf_train.get("strategy", "ddp"),
        precision=nf_train.get("precision", "bf16-mixed"),
        gradient_clip_val=nf_train.get("gradient_clip_val", 0.5),
        accumulate_grad_batches=nf_train.get("accumulate_grad_batches", 2),
        log_every_n_steps=nf_train.get("log_every_n_steps", 10),
        logger=None if args.fast_dev_run else WandbLogger(
            project=nf_train.get("wandb_project", "latent-pipeline-nf"),
            name=args.run_name, log_model=False, save_dir=ckpt_dir),
        callbacks=callbacks)

    resume = nf_train.get("resume_from")
    if resume is None and os.path.exists(os.path.join(ckpt_dir, "last.ckpt")):
        resume = os.path.join(ckpt_dir, "last.ckpt")
    trainer.fit(model, dm, ckpt_path=resume)


if __name__ == "__main__":
    main()
