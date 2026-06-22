"""NF branch: multi-suite Mgas -> (Omega_m, sigma_8) posterior.

Ported from ../upscaling/nf/module.py. Dataset is multi-suite (IllustrisTNG +
Astrid + SIMBA) and applies the SAME log1p+z-score Mgas normalization as the FM
branch (stats from cached/norm_latent.npz). Targets = first 2 cosmo params.
"""

from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from data import read_cube
from .encoder import ResNet3D
from .flow import ConditionalFlow


# ── augmentation ──────────────────────────────────────────────────────────────

class RandomRotateFlip3D:
    def __init__(self):
        self.axis_pairs = [(1, 2), (2, 3), (1, 3)]

    def __call__(self, vol):
        k = torch.randint(0, 4, (1,)).item()
        axes = self.axis_pairs[torch.randint(0, 3, (1,)).item()]
        vol = torch.rot90(vol, k, axes)
        for d in (1, 2, 3):
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, [d])
        return vol


class RandomPBCShift3D:
    def __call__(self, vol):
        D = vol.shape[1]
        shifts = tuple(torch.randint(0, D, (1,)).item() for _ in range(3))
        return torch.roll(vol, shifts, dims=(1, 2, 3))


# ── multi-suite dataset ───────────────────────────────────────────────────────

class MultiSuiteMgasDataset(Dataset):
    """Concatenates Mgas + (Omega_m, sigma_8) over suites. log1p+z-score Mgas.

    mgas_arrs : list[np.memmap]   per-suite Mgas
    cosmo_all : (N,2) float32     stacked cosmo targets
    flat      : list[(suite,local)]
    mgas_mean/std : field normalization (shared with FM branch)
    """

    def __init__(self, mgas_src, cosmo_all, flat, indices, mgas_mean, mgas_std,
                 augment=False, input_noise_std=0.0):
        self.mgas_src = mgas_src
        self.cosmo_all = cosmo_all
        self.flat = flat
        self.indices = np.asarray(indices)
        self.mgas_mean = float(mgas_mean)
        self.mgas_std = float(mgas_std)
        self.input_noise_std = input_noise_std
        self.aug = (RandomPBCShift3D(), RandomRotateFlip3D()) if augment else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        g = int(self.indices[idx])
        si, li = self.flat[g]
        v = read_cube(self.mgas_src[si][li])
        v = (np.log1p(v) - self.mgas_mean) / self.mgas_std
        vol = torch.from_numpy(v).unsqueeze(0)
        if self.aug is not None:
            for a in self.aug:
                vol = a(vol)
            if self.input_noise_std > 0:
                vol = vol + torch.randn_like(vol) * self.input_noise_std
        target = torch.from_numpy(self.cosmo_all[g].astype(np.float32).copy())
        return vol, target


class MultiSuiteNFDataModule(pl.LightningDataModule):
    def __init__(self, mgas_src, cosmo_all, flat, mgas_mean, mgas_std,
                 val_split=0.2, batch_size=8, num_workers=4, seed=42,
                 input_noise_std=0.0):
        super().__init__()
        self.mgas_src = mgas_src
        self.cosmo_all = cosmo_all
        self.flat = flat
        self.mgas_mean = mgas_mean
        self.mgas_std = mgas_std
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.input_noise_std = input_noise_std
        self.target_mean = None
        self.target_std = None

    def setup(self, stage=None):
        n = len(self.flat)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(n)
        n_val = max(1, int(n * self.val_split))
        va_idx, tr_idx = idx[:n_val], idx[n_val:]

        tr_targets = self.cosmo_all[tr_idx]
        self.target_mean = tr_targets.mean(0).astype(np.float32)
        std = tr_targets.std(0).astype(np.float32)
        self.target_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        print(f"NF targets over {len(tr_idx)} train: mean={self.target_mean}, std={self.target_std}")

        self.train_ds = MultiSuiteMgasDataset(
            self.mgas_src, self.cosmo_all, self.flat, tr_idx,
            self.mgas_mean, self.mgas_std, augment=True,
            input_noise_std=self.input_noise_std)
        self.val_ds = MultiSuiteMgasDataset(
            self.mgas_src, self.cosmo_all, self.flat, va_idx,
            self.mgas_mean, self.mgas_std, augment=False)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=(self.num_workers > 0),
                          prefetch_factor=4 if self.num_workers > 0 else None)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)


# ── velocity-NF data (v = Mgas_norm - Nbody_norm) ─────────────────────────────

class MultiSuiteVelocityDataset(Dataset):
    """Concatenates the FM 'velocity' field v = Mgas_norm - Nbody_norm + cosmo over
    suites. Both fields come from the PRE-NORMED FM cache (already log1p+z-scored);
    clamped to +/-clamp_val then differenced (matches data.CachedFMDataset). The signed
    v is z-scored by the SHARED velocity stats (data.velocity_stats) -- NO log1p.
    """

    def __init__(self, nbody_arrs, mgas_arrs, cosmo_all, flat, indices,
                 vel_mean, vel_std, clamp_val=10.0, augment=False, input_noise_std=0.0):
        self.nbody_arrs = nbody_arrs
        self.mgas_arrs = mgas_arrs
        self.cosmo_all = cosmo_all
        self.flat = flat
        self.indices = np.asarray(indices)
        self.vel_mean = float(vel_mean)
        self.vel_std = float(vel_std)
        self.clamp_val = clamp_val
        self.input_noise_std = input_noise_std
        self.aug = (RandomPBCShift3D(), RandomRotateFlip3D()) if augment else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        g = int(self.indices[idx])
        si, li = self.flat[g]
        nb = np.array(self.nbody_arrs[si][li], dtype=np.float32)
        mg = np.array(self.mgas_arrs[si][li], dtype=np.float32)
        if self.clamp_val is not None:
            c = float(self.clamp_val)
            np.clip(nb, -c, c, out=nb)
            np.clip(mg, -c, c, out=mg)
        v = (mg - nb - self.vel_mean) / self.vel_std
        vol = torch.from_numpy(v).unsqueeze(0)
        if self.aug is not None:
            for a in self.aug:
                vol = a(vol)
            if self.input_noise_std > 0:
                vol = vol + torch.randn_like(vol) * self.input_noise_std
        target = torch.from_numpy(self.cosmo_all[g].astype(np.float32).copy())
        return vol, target


class MultiSuiteVelocityDataModule(pl.LightningDataModule):
    """Same split/seed contract as MultiSuiteNFDataModule, but serves the velocity field
    built from the FM cache (nbody_arrs + mgas_arrs)."""

    def __init__(self, nbody_arrs, mgas_arrs, cosmo_all, flat, vel_mean, vel_std,
                 clamp_val=10.0, val_split=0.2, batch_size=8, num_workers=4, seed=42,
                 input_noise_std=0.0):
        super().__init__()
        self.nbody_arrs = nbody_arrs
        self.mgas_arrs = mgas_arrs
        self.cosmo_all = cosmo_all
        self.flat = flat
        self.vel_mean = vel_mean
        self.vel_std = vel_std
        self.clamp_val = clamp_val
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.input_noise_std = input_noise_std
        self.target_mean = None
        self.target_std = None

    def setup(self, stage=None):
        n = len(self.flat)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(n)
        n_val = max(1, int(n * self.val_split))
        va_idx, tr_idx = idx[:n_val], idx[n_val:]

        tr_targets = self.cosmo_all[tr_idx]
        self.target_mean = tr_targets.mean(0).astype(np.float32)
        std = tr_targets.std(0).astype(np.float32)
        self.target_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        print(f"velNF targets over {len(tr_idx)} train: mean={self.target_mean}, std={self.target_std}")

        self.train_ds = MultiSuiteVelocityDataset(
            self.nbody_arrs, self.mgas_arrs, self.cosmo_all, self.flat, tr_idx,
            self.vel_mean, self.vel_std, clamp_val=self.clamp_val, augment=True,
            input_noise_std=self.input_noise_std)
        self.val_ds = MultiSuiteVelocityDataset(
            self.nbody_arrs, self.mgas_arrs, self.cosmo_all, self.flat, va_idx,
            self.vel_mean, self.vel_std, clamp_val=self.clamp_val, augment=False)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=(self.num_workers > 0),
                          prefetch_factor=4 if self.num_workers > 0 else None)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)


# ── lightning module ──────────────────────────────────────────────────────────

class LitNFRegressor(pl.LightningModule):
    """ResNet3D + ConditionalFlow for Mgas -> cosmo posterior."""

    def __init__(
        self, lr_encoder=3e-4, lr_flow=1e-4, weight_decay=1e-4,
        base_channels=4, num_params=2, flow_hidden=128, flow_transforms=4,
        flow_type="nsf", flow_bins=8, dropout=0.15, summary_dim=None,
        summary_noise_std=0.0, target_noise_std=0.0, circular_padding=True,
        warmup_epochs=20, max_epochs=1500, flow_warmup_epochs=50,
        aux_loss_weight=0.5, aux_loss_decay=1.0, skip_nan_loss=True,
        target_mean=None, target_std=None, plot_every_n_epochs=5,
        plot_n_samples=64, plot_posterior_draws=200, param_names=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_plot_buf = None

        self.encoder = ResNet3D(in_ch=1, base=base_channels, dropout=dropout,
                                circular_padding=circular_padding, use_checkpoint=True)
        enc_dim = self.encoder.output_dim

        if summary_dim and summary_dim != enc_dim:
            self.summary_proj = nn.Linear(enc_dim, summary_dim)
            ctx_dim = summary_dim
        else:
            self.summary_proj = nn.Identity()
            ctx_dim = enc_dim

        self.aux_head = nn.Sequential(
            nn.Linear(ctx_dim, max(ctx_dim // 2, 2)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(max(ctx_dim // 2, 2), num_params))

        self.flow = ConditionalFlow(
            num_params=num_params, context_dim=ctx_dim, hidden_dim=flow_hidden,
            num_transforms=flow_transforms, flow_type=flow_type, bins=flow_bins,
            target_mean=target_mean, target_std=target_std)

        self.aux_weight = aux_loss_weight

    def forward(self, x):
        summary = self.summary_proj(self.encoder(x))
        if self.training and self.hparams.summary_noise_std > 0:
            summary = summary + torch.randn_like(summary) * self.hparams.summary_noise_std
        return summary, self.aux_head(summary)

    def _step(self, batch, stage):
        x, y = batch
        summary, aux_pred = self(x)

        y_flow = y
        if stage == "train" and self.hparams.target_noise_std > 0:
            y_flow = y + torch.randn_like(y) * self.hparams.target_noise_std
        log_prob = self.flow.log_prob(summary, y_flow)
        bad = torch.isnan(log_prob) | torch.isinf(log_prob)
        if bad.any():
            self.log(f"{stage}/nan_count", bad.sum().float(), sync_dist=True)
            log_prob = torch.where(bad, torch.full_like(log_prob, -100.0), log_prob)
        nll = -log_prob.mean()

        y_norm = (y - self.flow.target_mean) / self.flow.target_std
        aux_norm = (aux_pred - self.flow.target_mean) / self.flow.target_std
        aux_loss = F.mse_loss(aux_norm, y_norm)

        epoch = self.current_epoch or 0
        w = self.aux_weight * (self.hparams.aux_loss_decay ** epoch)
        nll_on = 1.0 if epoch >= self.hparams.flow_warmup_epochs else 0.0
        loss = nll_on * nll + w * aux_loss

        self.log(f"{stage}/nll", nll, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/aux", aux_loss, sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)

        if stage == "val":
            with torch.no_grad():
                pred_mean, pred_std = self.flow.get_posterior_stats(summary, num_samples=200)
                self.log("val/mae", (pred_mean - y).abs().mean(), prog_bar=True, sync_dist=True)
                self.log("val/post_std", pred_std.mean(), sync_dist=True)
            if getattr(self, "_trainer", None) is not None and getattr(self.trainer, "is_global_zero", True):
                cap = int(self.hparams.plot_n_samples)
                cur = self._val_plot_buf[0].shape[0] if self._val_plot_buf is not None else 0
                if cur < cap:
                    take = min(cap - cur, summary.shape[0])
                    s, tt = summary[:take].detach(), y[:take].detach()
                    self._val_plot_buf = (s, tt) if self._val_plot_buf is None else (
                        torch.cat([self._val_plot_buf[0], s]), torch.cat([self._val_plot_buf[1], tt]))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        if self.hparams.skip_nan_loss and not torch.isfinite(loss):
            self.log("nan_skip", 1.0, on_step=True, on_epoch=False)
            return None
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_validation_epoch_start(self):
        self._val_plot_buf = None

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            {"params": list(self.encoder.parameters()) + list(self.summary_proj.parameters()),
             "lr": self.hparams.lr_encoder, "name": "encoder"},
            {"params": self.aux_head.parameters(), "lr": self.hparams.lr_encoder, "name": "aux"},
            {"params": self.flow.parameters(), "lr": self.hparams.lr_flow, "name": "flow"},
        ], weight_decay=self.hparams.weight_decay)
        warmup_epochs = self.hparams.warmup_epochs
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs - warmup_epochs, eta_min=1e-6)
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, [warmup, cosine], milestones=[warmup_epochs])
        else:
            sched = cosine
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
