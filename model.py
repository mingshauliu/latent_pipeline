"""FM model for latent_pipeline: Nbody -> Mgas, conditioned on cosmology + an
8-dim latent encoded from Mgas.

Two pieces:
  - ClassicUNet  : 3D UNet velocity field (vendored from ../upscaling/model_classic.py),
                   extended so conditioning = time + cosmo(2) + latent(8) via FiLM.
                   PixelNorm (resolution-invariant) + circular padding + zero-init
                   FiLM/out_conv for a stable start despite large mean offsets.
  - GasEncoder   : SE-ResNet3D (optimised encoder design from ../upscaling/nf/encoder.py)
                   compressing Mgas (B,1,D,D,D) -> latent (B, latent_dim).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# shared bits
# ─────────────────────────────────────────────────────────────────────────────

class PixelNorm(nn.Module):
    def __init__(self, num_channels=None, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


def _make_norm(norm_type, num_channels):
    if norm_type == "pixel":
        return PixelNorm(num_channels)
    return nn.GroupNorm(1, num_channels)


def sinusoidal_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)
        # Zero-init: gamma=0 -> (1+gamma)*x = x, beta=0 at init (identity).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, c):
        s, b = self.proj(c).chunk(2, dim=1)
        return x * (1 + s.reshape(-1, x.size(1), 1, 1, 1)) + b.reshape(-1, x.size(1), 1, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# FM UNet
# ─────────────────────────────────────────────────────────────────────────────

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, circular, down=True, norm_type="pixel"):
        super().__init__()
        self.down = down
        self.pad_mode = "circular" if circular else "constant"
        self.norm1 = _make_norm(norm_type, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3)
        self.film1 = FiLM(cond_dim, out_ch)
        self.norm2 = _make_norm(norm_type, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3)
        self.film2 = FiLM(cond_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        if self.down:
            self.pool = nn.MaxPool3d(2)

    def forward(self, x, c):
        res = self.skip(x)
        h = self.act(self.norm1(x))
        h = self.film1(self.conv1(F.pad(h, [1]*6, mode=self.pad_mode)), c)
        h = self.act(self.norm2(h))
        h = self.film2(self.conv2(F.pad(h, [1]*6, mode=self.pad_mode)), c)
        h = self.act(h + res)
        if self.down:
            return h, self.pool(h)
        return h


class ClassicUNet(nn.Module):
    """3-level UNet velocity field. Conditioning = time + cosmo + latent via FiLM."""

    def __init__(self, in_channels=2, base_channels=128, out_channels=1,
                 cosmo_dim=2, latent_dim=8, circular_padding=True, norm_type="pixel"):
        super().__init__()
        bc = base_channels
        circ = circular_padding
        nt = norm_type
        self.pad_mode = "circular" if circ else "constant"
        self._ckpt = False
        cd = 128  # FiLM conditioning width

        self.time_mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
        self.cosmo_mlp = nn.Sequential(nn.Linear(cosmo_dim, 128), nn.SiLU(), nn.Linear(128, 64))
        # Fuse time(64) + cosmo(64) + latent(latent_dim) -> cd
        self.cond_fuse = nn.Sequential(
            nn.Linear(64 + 64 + latent_dim, cd * 2), nn.SiLU(), nn.Linear(cd * 2, cd))

        self.enc1 = UNetBlock(in_channels, bc, cd, circ, down=True, norm_type=nt)
        self.enc2 = UNetBlock(bc, bc, cd, circ, down=True, norm_type=nt)
        self.enc3 = UNetBlock(bc, 2*bc, cd, circ, down=True, norm_type=nt)

        self.bn_norm1 = _make_norm(nt, 2*bc)
        self.bn_conv1 = nn.Conv3d(2*bc, 2*bc, 3)
        self.bn_norm2 = _make_norm(nt, 2*bc)
        self.bn_conv2 = nn.Conv3d(2*bc, 2*bc, 3)
        self.bn_film = FiLM(cd, 2*bc)

        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up3_conv = nn.Conv3d(2*bc, 2*bc, 3)
        self.dec3 = UNetBlock(4*bc, 2*bc, cd, circ, down=False, norm_type=nt)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up2_conv = nn.Conv3d(2*bc, bc, 3)
        self.dec2 = UNetBlock(2*bc, bc, cd, circ, down=False, norm_type=nt)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up1_conv = nn.Conv3d(bc, bc // 2, 3)
        self.dec1 = UNetBlock(bc + bc // 2, bc // 2, cd, circ, down=False, norm_type=nt)

        self.out_conv = nn.Conv3d(bc // 2, out_channels, 1)
        # Zero-init final conv: v=0 at init -> loss = E[(x1-x0)^2] without
        # amplification from random output (target has large mean offset).
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def set_pad_mode(self, mode):
        self.pad_mode = mode
        for m in self.modules():
            if isinstance(m, UNetBlock):
                m.pad_mode = mode

    def enable_gradient_checkpointing(self):
        self._ckpt = True

    def _ckpt_call(self, module, *args):
        if self._ckpt and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x, t, cosmo, latent):
        c = self.cond_fuse(torch.cat([
            self.time_mlp(sinusoidal_embedding(t, 64)),
            self.cosmo_mlp(cosmo),
            latent,
        ], dim=1))

        s1, x = self._ckpt_call(self.enc1, x, c)
        s2, x = self._ckpt_call(self.enc2, x, c)
        s3, x = self._ckpt_call(self.enc3, x, c)

        x = F.silu(self.bn_norm1(x))
        x = self.bn_conv1(F.pad(x, [1]*6, mode=self.pad_mode))
        x = F.silu(self.bn_norm2(x))
        x = self.bn_conv2(F.pad(x, [1]*6, mode=self.pad_mode))
        x = self.bn_film(x, c)

        x = self.up3_conv(F.pad(self.up3(x), [1]*6, mode=self.pad_mode))
        x = self._ckpt_call(self.dec3, torch.cat([x, s3], 1), c)

        x = self.up2_conv(F.pad(self.up2(x), [1]*6, mode=self.pad_mode))
        x = self._ckpt_call(self.dec2, torch.cat([x, s2], 1), c)

        x = self.up1_conv(F.pad(self.up1(x), [1]*6, mode=self.pad_mode))
        x = self._ckpt_call(self.dec1, torch.cat([x, s1], 1), c)

        return self.out_conv(x)

    def forward_offload(self, x, t, cosmo, latent):
        """Same as forward but CPU-offloads skip connections s1, s2 between encode
        and decode — cuts peak GPU memory for large (e.g. 256^3) cubes at the cost
        of CPU<->GPU transfers. Vendored from ../upscaling/model_classic.py."""
        c = self.cond_fuse(torch.cat([
            self.time_mlp(sinusoidal_embedding(t, 64)),
            self.cosmo_mlp(cosmo),
            latent,
        ], dim=1))

        s1, x = self._ckpt_call(self.enc1, x, c)
        s1_cpu = s1.to("cpu"); del s1
        s2, x = self._ckpt_call(self.enc2, x, c)
        s2_cpu = s2.to("cpu"); del s2
        s3, x = self._ckpt_call(self.enc3, x, c)

        x = F.silu(self.bn_norm1(x))
        x = self.bn_conv1(F.pad(x, [1]*6, mode=self.pad_mode))
        x = F.silu(self.bn_norm2(x))
        x = self.bn_conv2(F.pad(x, [1]*6, mode=self.pad_mode))
        x = self.bn_film(x, c)

        x = self.up3_conv(F.pad(self.up3(x), [1]*6, mode=self.pad_mode))
        x = self._ckpt_call(self.dec3, torch.cat([x, s3], 1), c)
        del s3

        x = self.up2_conv(F.pad(self.up2(x), [1]*6, mode=self.pad_mode))
        s2 = s2_cpu.to(x.device); del s2_cpu
        x = self._ckpt_call(self.dec2, torch.cat([x, s2], 1), c)
        del s2

        x = self.up1_conv(F.pad(self.up1(x), [1]*6, mode=self.pad_mode))
        s1 = s1_cpu.to(x.device); del s1_cpu
        x = self._ckpt_call(self.dec1, torch.cat([x, s1], 1), c)
        del s1

        return self.out_conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Mgas -> 8-dim latent encoder (SE-ResNet3D, vendored design from upscaling/nf)
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock3D(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 8), bias=False), nn.SiLU(),
            nn.Linear(max(ch // reduction, 8), ch, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1, 1)


class _EncBlock(nn.Module):
    """Pre-activation residual + SE + strided downsample. Circular padding."""

    def __init__(self, in_ch, out_ch, dropout=0.1, pad_mode="circular"):
        super().__init__()
        self.pad_mode = pad_mode
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, bias=False)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout3d(p=dropout)
        self.residual_conv = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None
        self.down = nn.Conv3d(out_ch, out_ch, 3, stride=2, bias=False)
        self.se = SEBlock3D(out_ch)

    def _pad(self, h):
        return F.pad(h, (1,)*6, mode=self.pad_mode)

    def forward(self, x):
        residual = self.residual_conv(x) if self.residual_conv is not None else x
        h = self.conv1(self._pad(self.act(self.norm1(x))))
        h = self.act(self.norm2(h))
        h = self.conv2(self._pad(self.dropout(h)))
        h = self.se(h) + residual
        return self.down(self._pad(h))


class GasEncoder(nn.Module):
    """Mgas (B,1,D,D,D) -> latent (B, latent_dim). 3 strided SE-ResNet stages,
    global avg pool, LayerNorm, project to latent_dim.

    Two heads:
      - deterministic (default): project -> latent_dim, tanh-bounded to [-1, 1].
      - variational: project -> 2*latent_dim = (mu, logvar), NO tanh. forward
        returns (mu, logvar); the caller reparametrises z = mu + eps*exp(.5*logvar)
        and regularises with KL(q || N(0,I)). Makes infer latent_mode=sample
        (N(0,I)) valid by construction (prior == latent dist)."""

    def __init__(self, latent_dim=8, base=16, dropout=0.1,
                 circular_padding=True, use_checkpoint=True, variational=False,
                 in_channels=1):
        super().__init__()
        pad = "circular" if circular_padding else "constant"
        self.pad_mode = pad
        self.use_checkpoint = use_checkpoint
        self.variational = variational
        # in_channels=1 -> encode Mgas only (baseline). in_channels=2 -> encode the
        # stacked (Mgas, ne) target (multi-task variant); richer encode signal, may
        # resist latent collapse. The caller cats the fields before calling forward.
        self.stem = nn.Conv3d(in_channels, base, 3, bias=False)
        self.enc1 = _EncBlock(base, 2*base, dropout, pad)
        self.enc2 = _EncBlock(2*base, 4*base, dropout, pad)
        self.enc3 = _EncBlock(4*base, 8*base, dropout, pad)
        self.se = SEBlock3D(8*base)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.norm = nn.LayerNorm(8*base)
        self.proj = nn.Linear(8*base, latent_dim * 2 if variational else latent_dim)

    def _ckpt(self, module, x):
        if self.use_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x):
        x = self.stem(F.pad(x, (1,)*6, mode=self.pad_mode))
        x = self._ckpt(self.enc1, x)
        x = self._ckpt(self.enc2, x)
        x = self._ckpt(self.enc3, x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        x = self.norm(x)
        o = self.proj(x)
        if self.variational:
            mu, logvar = o.chunk(2, dim=-1)
            return mu, logvar
        return torch.tanh(o)
