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
    """Periodic-box augmentation for (B,C,D,H,W) cubes: random 90-deg rotation, axis
    flips, and (optional) PBC roll shift. The SAME transform is applied to every tensor
    so nbody/mgas stay voxel-aligned. All ops are symmetries of the periodic simulation
    box (circular padding in the net keeps them exact), so they multiply effective data
    without distorting the physics."""

    PAIRS = [(2, 3), (3, 4), (2, 4)]

    def __init__(self, pbc_shift=True):
        self.pbc_shift = pbc_shift

    def __call__(self, *tensors):
        k = torch.randint(0, 4, (1,)).item()
        axes = self.PAIRS[torch.randint(0, 3, (1,)).item()]
        tensors = tuple(torch.rot90(t, k, axes) for t in tensors)
        for d in (2, 3, 4):
            if torch.rand(1).item() < 0.5:
                tensors = tuple(torch.flip(t, [d]) for t in tensors)
        if self.pbc_shift:
            dims = (2, 3, 4)
            sizes = tensors[0].shape  # (B,C,D,H,W)
            shifts = [torch.randint(0, sizes[d], (1,)).item() for d in dims]
            tensors = tuple(torch.roll(t, shifts=shifts, dims=dims) for t in tensors)
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
        # Extra N-body VELOCITY input channel (separate model variant). When true the
        # dataloader yields a 4-tuple (nbody, mgas, cosmo, vel) and the UNet input is
        # cat([x_t, nbody, vel]) -> model.in_channels MUST be 3. Velocity is a STATIC
        # conditioning channel (NOT part of the x0/x1 flow). Default false = baseline.
        self.use_velocity = bool(d.get("use_velocity", False))
        # Extra Mgas-paired OUTPUT (target) channel: electron density (ne). When true the
        # loader yields ne (a 2nd target), the flow runs out_channels=2 = (Mgas, ne)
        # jointly (x0 = nbody repeated to 2ch + noise, x1 = cat([mgas, ne])), and the
        # latent encoder takes the stacked (Mgas, ne) -> in_channels=2. model.out_channels
        # MUST be 2 and model.in_channels = out_channels + 1 [+1 vel]. Default false =
        # single-channel Mgas baseline (fully backward compatible).
        self.use_ne = bool(d.get("use_ne", False))
        self.out_channels = m["out_channels"]

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
            in_channels=2 if self.use_ne else 1,   # encode (Mgas, ne) jointly when on
            latent_head=m.get("latent_head", "tanh"),  # raw|mlp drop tanh (encoder3D)
        )
        if t.get("gradient_checkpointing", False):
            self.net.enable_gradient_checkpointing()
        self.aug = RandomRotateFlip3D(pbc_shift=bool(t.get("aug_pbc_shift", True)))

        ema_cfg = t.get("ema") or {}
        self.ema_enabled = bool(ema_cfg.get("enabled", False))
        self.ema_decay = float(ema_cfg.get("decay", 0.9999))
        self.ema_warmup_steps = int(ema_cfg.get("warmup_steps", 0))
        self._ema_shadow = None
        self._ema_backup = None

        # ── velocity-NF cosmo-consistency aux loss (TRAINING-ONLY, opt-in) ──────
        # A frozen NF p(cosmo | v=Mgas-Nbody) judges the FM-predicted velocity v_pred:
        # aux = MSE(critic(v_pred), cosmo). Pressures the FM to keep cosmology readable
        # in its output. Fully gated: velocity_nf_ckpt=None -> OFF (backward compatible).
        # The critic is NEVER persisted in the FM ckpt and NEVER touched at sampling
        # time, so a ckpt trained with aux loads + samples with aux off and no NF file.
        self.vel_nf_ckpt = t.get("velocity_nf_ckpt")
        self.vel_aux_weight = float(t.get("vel_aux_weight", 0.0))
        self.vel_aux_warmup_epochs = int(t.get("vel_aux_warmup_epochs", 0))
        self._vel_critic = None
        if self.vel_nf_ckpt and self.vel_aux_weight > 0:
            self._build_vel_critic(t.get("velocity_stats", "cached/norm_velocity.npz"))

    # ── velocity-NF aux critic (frozen, training-only) ─────────────────────────
    def _build_vel_critic(self, vel_stats_path):
        """Load + freeze the velocity-NF critic and register the velocity-norm stats.
        Critic is a regular submodule (so Lightning moves it to the right device) but
        is stripped from the FM checkpoint in on_save_checkpoint, kept frozen + eval,
        and excluded from EMA (_fm_named_params) + the optimizer (requires_grad filter)."""
        from nf.module import LitNFRegressor
        critic = LitNFRegressor.load_from_checkpoint(self.vel_nf_ckpt, map_location="cpu")
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)
        self._vel_critic = critic
        z = np.load(vel_stats_path)
        # non-persistent: not written to the FM ckpt (single source of truth = npz)
        self.register_buffer("vel_mean", torch.tensor(float(z["vel_mean"])), persistent=False)
        self.register_buffer("vel_std", torch.tensor(float(z["vel_std"])), persistent=False)
        print(f"[FM aux] velocity-NF critic loaded: {self.vel_nf_ckpt} | "
              f"vel norm {float(z['vel_mean']):.4f}/{float(z['vel_std']):.4f} | "
              f"weight {self.vel_aux_weight} warmup {self.vel_aux_warmup_epochs}ep")

    def _aux_on(self):
        return self._vel_critic is not None and self.vel_aux_weight > 0

    def _vel_aux_loss(self, v_pred, cosmo):
        """MSE between the frozen critic's point cosmo estimate on v_pred and the true
        cosmo (both in the critic's normalized param space). Critic stays frozen/eval;
        gradient flows back through v_pred into the FM net only."""
        self._vel_critic.eval()
        v_in = (v_pred - self.vel_mean) / self.vel_std
        _, aux_pred = self._vel_critic(v_in)
        tm = self._vel_critic.flow.target_mean
        ts = self._vel_critic.flow.target_std
        return F.mse_loss((aux_pred - tm) / ts, (cosmo - tm) / ts)

    def _vel_aux_w(self):
        if self.vel_aux_warmup_epochs <= 0:
            return self.vel_aux_weight
        return self.vel_aux_weight * min(1.0, self.current_epoch / self.vel_aux_warmup_epochs)

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

    def _unpack(self, batch):
        """Decode the dataloader tuple by the use_ne / use_velocity flags (NOT by
        length): (nb, mg, [ne], cosmo, [vel]). Returns (nbody, mgas, ne, cosmo, vel)
        with ne / vel None when their flag is off."""
        nbody, mgas = batch[0], batch[1]
        i = 2
        ne = None
        if self.use_ne:
            ne = batch[i]; i += 1
        cosmo = batch[i]; i += 1
        vel = batch[i] if self.use_velocity else None
        return nbody, mgas, ne, cosmo, vel

    def _flow_targets(self, nbody, mgas, ne):
        """Build the flow endpoints. Single channel: x1=mgas, x0=nbody. Multi-task:
        x1=cat([mgas,ne]) and x0=nbody repeated to out_channels (each target channel
        flows from the N-body density start, encoder3D-style)."""
        if ne is None:
            return nbody, mgas
        x1 = torch.cat([mgas, ne], dim=1)                       # (B,2,...)
        base = nbody.expand(-1, x1.size(1), -1, -1, -1)         # repeat nbody -> 2ch start
        return base, x1

    def _step(self, batch, augment=False, sample_latent=False):
        nbody, mgas, ne, cosmo, vel = self._unpack(batch)
        if augment:
            tensors = [nbody, mgas] + ([ne] if ne is not None else []) \
                      + ([vel] if vel is not None else [])
            tensors = list(self.aug(*tensors))                  # shared transform -> aligned
            nbody, mgas = tensors[0], tensors[1]
            j = 2
            if ne is not None:
                ne = tensors[j]; j += 1
            if vel is not None:
                vel = tensors[j]
        B = nbody.size(0)
        enc_in = mgas if ne is None else torch.cat([mgas, ne], dim=1)
        latent, kl = self._encode_latent(enc_in, sample_latent)
        if self.time_sampling == "logitnormal":
            t = torch.sigmoid(torch.randn(B, device=nbody.device))
        else:
            t = torch.rand(B, device=nbody.device)
        base, x1 = self._flow_targets(nbody, mgas, ne)
        x0 = (base + torch.randn_like(base) * self.noise_std) if self.noise_std > 0 else base
        t_exp = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_exp) * x0 + t_exp * x1
        cond_in = [x_t, nbody] if vel is None else [x_t, nbody, vel]
        pred = self(torch.cat(cond_in, 1), t, cosmo, latent)
        return F.mse_loss(pred, x1 - x0), kl, pred, cosmo

    def training_step(self, batch, _):
        # sample the latent (reparam) so the KL is meaningful; guard on the RECON
        # term (KL is small, recon stays in the 0.02-0.7 regime the threshold targets).
        recon, kl, pred, cosmo = self._step(batch, augment=True, sample_latent=True)
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
        if self._aux_on():
            # critic judges the Mgas velocity channel (channel 0); ne channel ignored.
            aux = self._vel_aux_loss(pred[:, :1], cosmo)
            w = self._vel_aux_w()
            loss = loss + w * aux
            self.log("vel_aux", aux, prog_bar=True, on_step=False, on_epoch=True)
            self.log("vel_aux_w", w, on_step=False, on_epoch=True)
        self.log("train_loss", recon, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_enabled:
            self._ema_update()

    def validation_step(self, batch, batch_idx):
        # val recon uses the mean latent (no sampling) -> val_loss is comparable to
        # the deterministic baseline and across all sweep variants.
        loss, kl, _, _ = self._step(batch, sample_latent=False)
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
        # Never persist the frozen velocity-NF critic -> a ckpt trained WITH aux loads
        # + samples with aux OFF (velocity_nf_ckpt=None) and no NF file present.
        for k in [k for k in sd if k.startswith("_vel_critic.")]:
            del sd[k]
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
            buf[:, :x.shape[1]] = x   # flow channels (1 = Mgas, or 2 = Mgas+ne)
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
        nbody, mgas, ne, cosmo, vel = self._unpack(batch)
        nb32, cosmo32 = nbody.float(), cosmo.float()
        B, dev = nb32.size(0), nb32.device
        oc = self.out_channels
        # oracle latent = encode the TRUE target ((Mgas) or (Mgas,ne)); variational -> mu.
        enc_in = mgas.float() if ne is None else torch.cat([mgas.float(), ne.float()], dim=1)
        latent, _ = self._encode_latent(enc_in, sample_latent=False)
        base = nb32 if oc == 1 else nb32.expand(-1, oc, -1, -1, -1)
        x0 = base + torch.randn_like(base) * self.noise_std if self.noise_std > 0 else base.clone()
        cw = oc + (2 if vel is not None else 1)   # flow(oc) + nbody(1) [+ vel(1)]
        buf = torch.empty(B, cw, *nb32.shape[2:], device=dev, dtype=torch.float32)
        buf[:, oc:oc + 1] = nb32
        if vel is not None:
            buf[:, oc + 1:oc + 2] = vel.float()
        t_span = torch.linspace(0.0, 1.0, self.xcorr_steps + 1, device=dev)
        with torch.amp.autocast("cuda", enabled=False):
            x = odeint(self._ode_func(cosmo32, latent.float(), buf), x0, t_span,
                       method=self.xcorr_method,
                       **self._odeint_kwargs(self.xcorr_method, self.xcorr_steps,
                                             self.xcorr_rtol, self.xcorr_atol))[-1]
        # acceptance metric is on the Mgas channel (channel 0)
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
               rtol=1e-4, atol=1e-4, offload_skips=False, vel=None):
        """Integrate Nbody -> Mgas. `latent` is (B, latent_dim) supplied by caller
        (mean=zeros, sampled from a prior, or encoded from a reference cube).
        `vel` (B,1,D,D,D) = the N-body velocity conditioning channel (required when the
        model was trained with use_velocity; ignored otherwise).
        offload_skips: CPU-offload skip connections (use for large 256^3 cubes)."""
        self.eval()
        B, dev = nbody.size(0), nbody.device
        oc = self.out_channels   # 1 = Mgas (returns (B,1,...)); 2 = (Mgas, ne)
        base = nbody if oc == 1 else nbody.expand(-1, oc, -1, -1, -1)
        x0 = base + torch.randn_like(base) * self.noise_std if self.noise_std > 0 else base.clone()
        cw = oc + (2 if vel is not None else 1)   # flow(oc) + nbody(1) [+ vel(1)]
        buf = torch.empty(B, cw, *nbody.shape[2:], device=dev, dtype=nbody.dtype)
        buf[:, oc:oc + 1] = nbody
        if vel is not None:
            buf[:, oc + 1:oc + 2] = vel.to(buf.dtype)
        ADAPTIVE = {'dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun'}
        t_span = (torch.tensor([0.0, 1.0], device=dev) if method in ADAPTIVE
                  else torch.linspace(0.0, 1.0, num_steps + 1, device=dev))
        with torch.no_grad():
            traj = odeint(self._ode_func(cosmo, latent, buf, offload_skips), x0, t_span,
                          method=method, **self._odeint_kwargs(method, num_steps, rtol, atol))
        return traj[-1]

    def configure_optimizers(self):
        # filter out the frozen velocity-NF critic params (requires_grad=False)
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
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
