"""3D ResNet encoder for Mgas -> summary vector (vendored from ../upscaling/nf).

No FiLM conditioning — encoder is unconditional and only sees the gas field.
Pre-activation residual blocks, GroupNorm, Squeeze-Excitation, strided-conv
downsampling with circular padding for periodic boxes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SEBlock3D(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 8), bias=False),
            nn.SiLU(),
            nn.Linear(max(ch // reduction, 8), ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class EncoderBlock(nn.Module):
    """Pre-activation residual + SE + strided downsample. Circular padding."""

    def __init__(self, in_ch, out_ch, dropout=0.1, pad_mode="circular"):
        super().__init__()
        self.pad_mode = pad_mode
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=0, bias=False)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=0, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout3d(p=dropout)
        self.residual_conv = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=0, bias=False)
        self.se = SEBlock3D(out_ch)

    def _pad(self, h):
        return F.pad(h, (1, 1, 1, 1, 1, 1), mode=self.pad_mode)

    def forward(self, x):
        residual = self.residual_conv(x) if self.residual_conv is not None else x
        h = self.act(self.norm1(x))
        h = self.conv1(self._pad(h))
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(self._pad(h))
        h = self.se(h) + residual
        return self.down(self._pad(h))


class ResidualBlock(nn.Module):
    """Pre-activation residual block, no downsample."""

    def __init__(self, ch, expansion=2, dropout=0.1, pad_mode="circular"):
        super().__init__()
        mid = ch * expansion
        self.pad_mode = pad_mode
        self.norm1 = nn.GroupNorm(1, ch)
        self.conv1 = nn.Conv3d(ch, mid, 3, padding=0, bias=False)
        self.norm2 = nn.GroupNorm(1, mid)
        self.conv2 = nn.Conv3d(mid, ch, 3, padding=0, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout3d(p=dropout)

    def _pad(self, h):
        return F.pad(h, (1, 1, 1, 1, 1, 1), mode=self.pad_mode)

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(self._pad(h))
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(self._pad(h))
        return x + h


class ResNet3D(nn.Module):
    """3D ResNet encoder: Mgas (B,1,D,D,D) -> summary (B, 8*base)."""

    def __init__(self, in_ch=1, base=16, dropout=0.2,
                 circular_padding=True, use_checkpoint=True):
        super().__init__()
        pad = "circular" if circular_padding else "constant"
        self.pad_mode = pad
        self.use_checkpoint = use_checkpoint
        self.stem = nn.Conv3d(in_ch, base, 3, padding=0, bias=False)
        self.enc1 = EncoderBlock(base, 2 * base, dropout=dropout, pad_mode=pad)
        self.enc2 = EncoderBlock(2 * base, 4 * base, dropout=dropout, pad_mode=pad)
        self.enc3 = EncoderBlock(4 * base, 8 * base, dropout=dropout, pad_mode=pad)
        self.bottleneck_res1 = ResidualBlock(8 * base, expansion=2, dropout=dropout, pad_mode=pad)
        self.bottleneck_res2 = ResidualBlock(8 * base, expansion=2, dropout=dropout, pad_mode=pad)
        self.bottleneck_se = SEBlock3D(8 * base)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.output_norm = nn.LayerNorm(8 * base)

    @property
    def output_dim(self):
        return self.output_norm.normalized_shape[0]

    def _ckpt(self, module, x):
        if self.use_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode=self.pad_mode)
        x = self.stem(x)
        x = self._ckpt(self.enc1, x)
        x = self._ckpt(self.enc2, x)
        x = self._ckpt(self.enc3, x)
        x = self._ckpt(self.bottleneck_res1, x)
        x = self._ckpt(self.bottleneck_res2, x)
        x = self.bottleneck_se(x)
        x = self.pool(x).flatten(1)
        return self.output_norm(x)
