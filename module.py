"""Lightning module for the FM branch: Nbody -> Mgas with an 8-dim Mgas latent.

Ported from ../upscaling/train.py::FlowMatchingModel, extended with a GasEncoder
(Mgas -> latent) whose output conditions the UNet alongside cosmology. EMA covers
both the UNet and the encoder.
"""

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchdiffeq import odeint

from model import ClassicUNet, GasEncoder

try:
    import Pk_library as PKL
    HAS_PKL = True
except ImportError:
    HAS_PKL = False


# ── augmentation ──────────────────────────────────────────────────────────────

class RandomRotateFlip3D:
    PAIRS = [(2, 3), (3, 4), (2, 4)]

    def __call__(self, *tensors):
        k = torch.randint(0, 4, (1,)).item()
        axes = self.PAIRS[torch.randint(0, 3, (1,)).item()]
        tensors = tuple(torch.rot90(t, k, axes) for t in tensors)
        for d in (2, 3, 4):
            if torch.rand(1).item() < 0.5:
                tensors = tuple(torch.flip(t, [d]) for t in tensors)
        return tensors


# ── metrics ───────────────────────────────────────────────────────────────────

def xcorr_metric(d1, d2, box_size):
    d1 = np.ascontiguousarray((d1 - d1.mean()) / d1.std(), dtype=np.float32)
    d2 = np.ascontiguousarray((d2 - d2.mean()) / d2.std(), dtype=np.float32)
    Pk = PKL.XPk([d1, d2], box_size, 0, MAS=["CIC", "CIC"], threads=1)
    k = Pk.k1D
    denom = np.sqrt(np.clip(Pk.Pk1D[:, 0] * Pk.Pk1D[:, 1], 1e-30, None))
    xpk = Pk.PkX1D[:, 0] / denom
    m = (k <= 15) & np.isfinite(xpk)
    if m.sum() < 2:
        return float("nan")
    return float(np.trapz(xpk[m], k[m]) / (k[m].max() - k[m].min()))


def pk_ratio_metric(synth, true, box_size, k_max=15.0):
    Pk_s = PKL.Pk(synth.astype(np.float32), box_size, MAS="CIC", threads=1)
    Pk_t = PKL.Pk(true.astype(np.float32), box_size, MAS="CIC", threads=1)
    k = Pk_s.k3D
    m = k <= k_max
    ratio = Pk_s.Pk[:, 0] / (Pk_t.Pk[:, 0] + 1e-30)
    return float(np.nanmean(ratio[m]))


# ── lightning module ──────────────────────────────────────────────────────────

class FlowMatchingModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        m, t, d = cfg["model"], cfg["training"], cfg["data"]
        self.lr = t["lr"]
        self.wd = t["weight_decay"]
        self.noise_std = t["noise_std"]
        # Time sampling for the FM interpolant. 'uniform' = t~U(0,1) (original).
        # 'logitnormal' = t=sigmoid(N(0,1)) — concentrates t near 0.5 (SD3/EDM), where
        # the velocity field is hardest; the encoder3D recipe that reaches xcorr ~0.9.
        self.time_sampling = t.get("time_sampling", "uniform")
        self.scheduler = t.get("scheduler", "cosine")
        self.warmup_epochs = t.get("warmup_epochs", 0)
        self.max_epochs = t["max_epochs"]
        self.xcorr_every = t["xcorr_every_n_epochs"]
        self.xcorr_steps = t["xcorr_num_steps"]
        self.xcorr_method = t.get("xcorr_method", "euler")
        self.xcorr_rtol = t.get("xcorr_rtol", 1e-4)
        self.xcorr_atol = t.get("xcorr_atol", 1e-4)
        self.skip_nan_loss = bool(t.get("skip_nan_loss", True))
        # Skip finite-but-huge loss batches (heavy-tail Nbody outlier cubes drive
        # per-batch MSE to ~1e4 and nuke the converged model). Normal loss is
        # 0.02-0.7, so a threshold ~5 is well clear. None disables the guard.
        self.spike_thresh = t.get("loss_spike_thresh", 5.0)
        crop = d.get("crop_size")
        res = d["resolution"]
        self.box_size = d["box_size"] * (crop / res) if crop and crop < res else d["box_size"]
        self.latent_dim = m.get("latent_dim", 8)

        # Variational (beta-VAE) latent: GasEncoder -> (mu, logvar), z reparametrised,
        # KL(q||N(0,I)) added to the recon loss. beta warms up linearly; per-dim
        # free-bits floor keeps individual dims from fully collapsing.
        self.variational = bool(m.get("variational", False))
        self.kl_beta = float(t.get("kl_beta", 1.0e-3))
        self.kl_warmup_epochs = int(t.get("kl_warmup_epochs", 0))
        self.kl_free_bits = float(t.get("kl_free_bits", 0.0))

        self.net = ClassicUNet(
            in_channels=m["in_channels"],
            base_channels=m["base_channels"],
            out_channels=m["out_channels"],
            cosmo_dim=m.get("cosmo_dim", 2),
            latent_dim=self.latent_dim,
            circular_padding=m["circular_padding"],
            norm_type=m.get("norm_type", "pixel"),
        )
        self.gas_encoder = GasEncoder(
            latent_dim=self.latent_dim,
            base=m.get("encoder_base", 16),
            dropout=m.get("encoder_dropout", 0.1),
            circular_padding=m["circular_padding"],
            variational=self.variational,
        )
        if t.get("gradient_checkpointing", False):
            self.net.enable_gradient_checkpointing()
        self.aug = RandomRotateFlip3D()

        ema_cfg = t.get("ema") or {}
        self.ema_enabled = bool(ema_cfg.get("enabled", False))
        self.ema_decay = float(ema_cfg.get("decay", 0.9999))
        self.ema_warmup_steps = int(ema_cfg.get("warmup_steps", 0))
        self._ema_shadow = None
        self._ema_backup = None

    # latent + UNet params get EMA, keyed to match checkpoint state_dict.
    def _fm_named_params(self):
        for n, p in self.net.named_parameters():
            yield f"net.{n}", p
        for n, p in self.gas_encoder.named_parameters():
            yield f"gas_encoder.{n}", p

    def forward(self, x, t, cosmo, latent):
        return self.net(x, t, cosmo, latent)

    def _encode_latent(self, mgas, sample_latent):
        """Returns (latent, kl). Deterministic -> (tanh latent, None).
        Variational -> reparametrised z (mu if not sample_latent) + scalar KL."""
        enc = self.gas_encoder(mgas)
        if not self.variational:
            return enc, None
        mu, logvar = enc
        if sample_latent:
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            latent = mu
        return latent, self._kl(mu, logvar)

    def _kl(self, mu, logvar):
        # per-dim KL(q||N(0,I)) averaged over the batch, free-bits floor per dim,
        # then summed over dims.
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
        kld = kld.mean(0)                                      # (D,)
        if self.kl_free_bits > 0:
            kld = torch.clamp(kld, min=self.kl_free_bits)
        return kld.sum()

    def _beta(self):
        if self.kl_warmup_epochs <= 0:
            return self.kl_beta
        return self.kl_beta * min(1.0, self.current_epoch / self.kl_warmup_epochs)

    def _step(self, batch, augment=False, sample_latent=False):
        nbody, mgas, cosmo = batch
        if augment:
            nbody, mgas = self.aug(nbody, mgas)
        B = nbody.size(0)
        latent, kl = self._encode_latent(mgas, sample_latent)
        if self.time_sampling == "logitnormal":
            t = torch.sigmoid(torch.randn(B, device=nbody.device))
        else:
            t = torch.rand(B, device=nbody.device)
        x0 = (nbody + torch.randn_like(nbody) * self.noise_std) if self.noise_std > 0 else nbody
        x1 = mgas
        t_exp = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_exp) * x0 + t_exp * x1
        pred = self(torch.cat([x_t, nbody], 1), t, cosmo, latent)
        return F.mse_loss(pred, x1 - x0), kl

    def training_step(self, batch, _):
        # sample the latent (reparam) so the KL is meaningful; guard on the RECON
        # term (KL is small, recon stays in the 0.02-0.7 regime the threshold targets).
        recon, kl = self._step(batch, augment=True, sample_latent=True)
        if self.skip_nan_loss and not torch.isfinite(recon):
            self.log("nan_skip", 1.0, on_step=True, on_epoch=False)
            return None
        if self.spike_thresh is not None and recon.detach() > self.spike_thresh:
            self.log("spike_skip", 1.0, on_step=True, on_epoch=False)
            return None
        loss = recon
        if self.variational:
            beta = self._beta()
            loss = recon + beta * kl
            self.log("kl", kl, prog_bar=True, on_step=False, on_epoch=True)
            self.log("kl_beta", beta, on_step=False, on_epoch=True)
        self.log("train_loss", recon, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_enabled:
            self._ema_update()

    def validation_step(self, batch, batch_idx):
        # val recon uses the mean latent (no sampling) -> val_loss is comparable to
        # the deterministic baseline and across all sweep variants.
        loss, kl = self._step(batch, sample_latent=False)
        if self.variational and kl is not None:
            self.log("val_kl", kl, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_epoch=True, prog_bar=True)
        is_xcorr_epoch = (not self.trainer.sanity_checking
                          and self.xcorr_every > 0
                          and self.current_epoch % self.xcorr_every == 0)
        if HAS_PKL and is_xcorr_epoch and batch_idx == 0:
            self._log_xcorr(batch)
        return loss

    def on_validation_epoch_start(self):
        if self.ema_enabled and self._ema_shadow is not None:
            self._ema_swap_in()

    def on_validation_epoch_end(self):
        if self.ema_enabled and self._ema_backup is not None:
            self._ema_swap_out()

    # ── EMA ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ema_update(self):
        if self.global_step < self.ema_warmup_steps:
            return
        if self._ema_shadow is None:
            self._ema_shadow = {n: p.detach().clone().float()
                                for n, p in self._fm_named_params() if p.requires_grad}
            return
        for n, p in self._fm_named_params():
            if not p.requires_grad:
                continue
            if self._ema_shadow[n].device != p.device:
                self._ema_shadow[n] = self._ema_shadow[n].to(p.device)
            self._ema_shadow[n].mul_(self.ema_decay).add_(p.detach().float(),
                                                          alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def _ema_swap_in(self):
        self._ema_backup = {}
        for n, p in self._fm_named_params():
            if n in self._ema_shadow:
                if self._ema_shadow[n].device != p.device:
                    self._ema_shadow[n] = self._ema_shadow[n].to(p.device)
                self._ema_backup[n] = p.detach().clone()
                p.data.copy_(self._ema_shadow[n].to(p.dtype))

    @torch.no_grad()
    def _ema_swap_out(self):
        backup = dict(self._ema_backup)
        for n, p in self._fm_named_params():
            if n in backup:
                p.data.copy_(backup[n])
        self._ema_backup = None

    def on_save_checkpoint(self, checkpoint):
        sd = checkpoint["state_dict"]
        if self.ema_enabled and self._ema_shadow is not None:
            checkpoint["live_state_dict"] = {
                k: v.detach().cpu().clone()
                for k, v in sd.items()
                if k.startswith("net.") or k.startswith("gas_encoder.")
            }
            for n, ema_t in self._ema_shadow.items():
                if n in sd:
                    sd[n] = ema_t.to(sd[n].dtype).detach().cpu().clone()
            checkpoint["ema_shadow"] = {n: t.detach().cpu() for n, t in self._ema_shadow.items()}

    def on_load_checkpoint(self, checkpoint):
        if self.ema_enabled and "ema_shadow" in checkpoint:
            self._ema_shadow = {n: t for n, t in checkpoint["ema_shadow"].items()}
        try:
            is_resume = self.trainer is not None and \
                getattr(self.trainer, "state", None) is not None and \
                getattr(self.trainer.state, "fn", None) == "fit"
        except RuntimeError:
            is_resume = False
        if is_resume and "live_state_dict" in checkpoint:
            for k, v in checkpoint["live_state_dict"].items():
                checkpoint["state_dict"][k] = v

    # ── ODE / sampling ─────────────────────────────────────────────────────────
    def _ode_func(self, cosmo, latent, buf, offload_skips=False):
        B = buf.size(0)
        fwd = self.net.forward_offload if offload_skips else self.net
        def f(t, x):
            buf[:, 0:1] = x
            return fwd(buf, t.expand(B), cosmo, latent)
        return f

    @staticmethod
    def _odeint_kwargs(method, num_steps, rtol, atol):
        ADAPTIVE = {'dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun'}
        if method in ADAPTIVE:
            return {'rtol': rtol, 'atol': atol}
        return {'options': {'step_size': 1.0 / num_steps}}

    @torch.no_grad()
    def _log_xcorr(self, batch):
        # Oracle latent: encode the TRUE Mgas (upper bound on FM transport given
        # the right latent). Inference without truth uses latent_mode in infer.py.
        nbody, mgas, cosmo = batch
        nb32, cosmo32 = nbody.float(), cosmo.float()
        B, dev = nb32.size(0), nb32.device
        # oracle latent = encode the TRUE Mgas; variational uses the mean (mu).
        latent, _ = self._encode_latent(mgas.float(), sample_latent=False)
        x0 = nb32 + torch.randn_like(nb32) * self.noise_std if self.noise_std > 0 else nb32.clone()
        buf = torch.empty(B, 2, *nb32.shape[2:], device=dev, dtype=torch.float32)
        buf[:, 1:2] = nb32
        t_span = torch.linspace(0.0, 1.0, self.xcorr_steps + 1, device=dev)
        with torch.amp.autocast("cuda", enabled=False):
            x = odeint(self._ode_func(cosmo32, latent.float(), buf), x0, t_span,
                       method=self.xcorr_method,
                       **self._odeint_kwargs(self.xcorr_method, self.xcorr_steps,
                                             self.xcorr_rtol, self.xcorr_atol))[-1]
        d1 = x[0, 0].cpu().numpy()
        d2 = mgas[0, 0].float().cpu().numpy()
        if (np.std(d1) < 1e-8 or np.std(d2) < 1e-8
                or not np.isfinite(d1).all() or not np.isfinite(d2).all()):
            return
        try:
            val = xcorr_metric(d1, d2, self.box_size)
            if np.isfinite(val):
                self.log("xcorr", val, prog_bar=True, on_step=False, on_epoch=True)
        except Exception as e:
            print(f"  xcorr failed: {e}")
        try:
            pkr = pk_ratio_metric(d1, d2, self.box_size)
            if np.isfinite(pkr):
                self.log("pk_ratio", pkr, prog_bar=True, on_step=False, on_epoch=True)
        except Exception as e:
            print(f"  pk_ratio failed: {e}")

    def sample(self, nbody, cosmo, latent, num_steps=100, method='euler',
               rtol=1e-4, atol=1e-4, offload_skips=False):
        """Integrate Nbody -> Mgas. `latent` is (B, latent_dim) supplied by caller
        (mean=zeros, sampled from a prior, or encoded from a reference cube).
        offload_skips: CPU-offload skip connections (use for large 256^3 cubes)."""
        self.eval()
        B, dev = nbody.size(0), nbody.device
        x0 = nbody + torch.randn_like(nbody) * self.noise_std if self.noise_std > 0 else nbody.clone()
        buf = torch.empty(B, 2, *nbody.shape[2:], device=dev, dtype=nbody.dtype)
        buf[:, 1:2] = nbody
        ADAPTIVE = {'dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun'}
        t_span = (torch.tensor([0.0, 1.0], device=dev) if method in ADAPTIVE
                  else torch.linspace(0.0, 1.0, num_steps + 1, device=dev))
        with torch.no_grad():
            traj = odeint(self._ode_func(cosmo, latent, buf, offload_skips), x0, t_span,
                          method=method, **self._odeint_kwargs(method, num_steps, rtol, atol))
        return traj[-1]

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        t = dict(self.hparams["training"])
        eta_min = float(t.get("eta_min", 1e-6))

        if self.scheduler == "cosine":
            # T_max defaults to the full run (current behaviour: lr barely anneals
            # over 2000 epochs). Set cosine_t_max << max_epochs for an AGGRESSIVE
            # anneal that actually drives lr -> eta_min to break the high-lr plateau.
            t_max = int(t.get("cosine_t_max", self.max_epochs - self.warmup_epochs))
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=t_max, eta_min=eta_min)
            if self.warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.01, total_iters=self.warmup_epochs)
                sched = torch.optim.lr_scheduler.SequentialLR(
                    opt, [warmup, cosine], milestones=[self.warmup_epochs])
            else:
                sched = cosine
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        if self.scheduler == "cosine_restarts":
            # SGDR: periodic re-heat to escape the basin, then anneal each cycle.
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=int(t.get("restart_t0", 50)),
                T_mult=int(t.get("restart_tmult", 2)), eta_min=eta_min)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        if self.scheduler == "onecycle":
            # Aggressive: ramp to max_lr then anneal hard to ~lr/div_final.
            total = int(self.trainer.estimated_stepping_batches)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=float(t.get("onecycle_max_lr", self.lr)),
                total_steps=total, pct_start=float(t.get("onecycle_pct_start", 0.1)),
                div_factor=float(t.get("onecycle_div_factor", 25.0)),
                final_div_factor=float(t.get("onecycle_final_div", 1e4)))
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        # plateau (adaptive): drop lr when val_loss stalls.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=float(t.get("plateau_factor", 0.5)),
            patience=int(t.get("plateau_patience", 15)), min_lr=eta_min)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
