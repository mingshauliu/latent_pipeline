"""Callable NF API (vendored from ../upscaling/nf/predict.py)."""

import numpy as np
import torch


def load_nf(ckpt_path, device="cuda"):
    from .module import LitNFRegressor
    model = LitNFRegressor.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def predict_with_uncertainty(model, dataloader, n_samples=2000, device="cuda"):
    """Run NF over a DataLoader yielding (vol, target). Returns
    (y_true, y_mean, y_std, aux_pred), each (N, num_params)."""
    yt, ym, ys, ap = [], [], [], []
    for batch in dataloader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        summary, aux = model(x)
        samples = model.flow.sample(summary, num_samples=n_samples).permute(1, 0, 2)
        ym.append(samples.mean(1).cpu().numpy())
        ys.append(samples.std(1).cpu().numpy())
        yt.append(y.numpy())
        ap.append(aux.cpu().numpy())
    return (np.concatenate(yt), np.concatenate(ym),
            np.concatenate(ys), np.concatenate(ap))
