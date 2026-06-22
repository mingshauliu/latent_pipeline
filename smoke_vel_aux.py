"""GPU smoke for the velocity-NF aux-loss wiring (Part 2).

Validates, at production size (128^3) on GPU, the full modular/backward-compatible
contract WITHOUT needing a trained velocity-NF (uses a fresh random critic just to
exercise the plumbing):

  1. velocity dataset: v = Mgas_norm - Nbody_norm from the FM cache (finite, signed).
  2. FM builds a frozen critic, aux loss is finite, grad flows to net NOT critic.
  3. critic is stripped from the FM checkpoint (on_save_checkpoint).
  4. an aux-trained ckpt reloads with aux OFF (no critic, no npz) and samples cleanly.

    python smoke_vel_aux.py --device cuda          # full 128^3
    python smoke_vel_aux.py --device cpu --tiny     # quick CPU plumbing check

Real end-to-end is separate: train_velocity_nf.sbatch (real velocity-NF) -> set
config training.velocity_nf_ckpt -> submit FM.
"""

import argparse, copy, os, tempfile, warnings
import numpy as np, torch, yaml, pytorch_lightning as pl
warnings.filterwarnings("ignore")

from nf.module import LitNFRegressor, MultiSuiteVelocityDataset
from module import FlowMatchingModel
import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true", help="16^3 + small FM (CPU plumbing)")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = yaml.safe_load(open(args.config))
    dcfg = cfg["data"]; clamp = dcfg.get("clamp_val", 10.0)

    # ── 1. velocity dataset from the real FM cache ─────────────────────────────
    nb, mg, co, flat = data.load_cache_pool(dcfg)
    vm, vs = data.compute_velocity_stats(nb, mg, flat, n_sample=8, clamp_val=clamp)
    ds = MultiSuiteVelocityDataset(nb, mg, co, flat, list(range(2)), vm, vs,
                                   clamp_val=clamp, augment=True)
    vol, tgt = ds[0]
    assert torch.isfinite(vol).all() and (vol < 0).any(), "velocity field bad"
    print(f"[1] velocity ds OK  shape {tuple(vol.shape)}  vel norm {vm:.3f}/{vs:.3f}  "
          f"min {float(vol.min()):.2f} max {float(vol.max()):.2f}")

    # ── critic (fresh, random) + stats npz ─────────────────────────────────────
    tmp = tempfile.mkdtemp()
    crit = LitNFRegressor(base_channels=cfg["nf"]["model"]["base_channels"], num_params=2,
                          flow_hidden=cfg["nf"]["model"].get("flow_hidden", 128),
                          flow_transforms=cfg["nf"]["model"].get("flow_transforms", 4),
                          summary_dim=cfg["nf"]["model"].get("summary_dim", 6),
                          target_mean=np.float32([0.3, 0.8]), target_std=np.float32([0.1, 0.1]))
    ck = os.path.join(tmp, "vel.ckpt")
    torch.save({"state_dict": crit.state_dict(), "hyper_parameters": dict(crit.hparams),
                "pytorch-lightning_version": pl.__version__}, ck)
    npz = os.path.join(tmp, "vel.npz"); np.savez(npz, vel_mean=np.float32(vm), vel_std=np.float32(vs))

    if args.tiny:
        cfg["model"]["base_channels"] = 8; cfg["model"]["encoder_base"] = 4; D = 16
    else:
        D = dcfg["resolution"]
    t = cfg["training"]; t["ema"] = {"enabled": False}; t["resume_from"] = None; t["init_from"] = None
    t["velocity_nf_ckpt"] = ck; t["vel_aux_weight"] = 0.05; t["vel_aux_warmup_epochs"] = 0
    t["velocity_stats"] = npz

    fm = FlowMatchingModel(cfg).to(dev)
    B = 2
    nbt = torch.randn(B, 1, D, D, D, device=dev); mgt = torch.randn(B, 1, D, D, D, device=dev)
    cot = torch.randn(B, 2, device=dev)
    fm.train()
    recon, kl, pred, cosmo = fm._step((nbt, mgt, cot), augment=True, sample_latent=True)
    aux = fm._vel_aux_loss(pred, cosmo); loss = recon + 0.05 * aux
    assert torch.isfinite(loss)
    loss.backward()
    gnet = any(p.grad is not None and p.grad.abs().sum() > 0 for p in fm.net.parameters())
    gcrit = any(p.grad is not None for p in fm._vel_critic.parameters())
    assert gnet and not gcrit, "grad routing wrong"
    print(f"[2] FM aux OK  recon {float(recon):.3f}  aux {float(aux):.3f}  grad->net {gnet} grad->critic {gcrit}")

    chk = {"state_dict": copy.deepcopy(fm.state_dict())}; fm.on_save_checkpoint(chk)
    assert not any(k.startswith("_vel_critic.") for k in chk["state_dict"])
    assert not any("vel_mean" in k or "vel_std" in k for k in chk["state_dict"])
    print("[3] checkpoint clean OK  (no critic / no vel stats persisted)")

    cfg2 = copy.deepcopy(cfg); cfg2["training"]["velocity_nf_ckpt"] = None; cfg2["training"]["vel_aux_weight"] = 0.0
    fm2 = FlowMatchingModel(cfg2).to(dev)
    miss, unexp = fm2.load_state_dict(chk["state_dict"], strict=False)
    assert not fm2._aux_on() and len(unexp) == 0
    fm2.eval()
    with torch.no_grad():
        s = fm2.sample(torch.randn(B, 1, D, D, D, device=dev), torch.randn(B, 2, device=dev),
                       torch.zeros(B, fm2.latent_dim, device=dev), num_steps=2, method="euler")
    assert torch.isfinite(s).all()
    print(f"[4] backward-compat OK  aux_off, reload unexpected {len(unexp)}, sample {tuple(s.shape)} finite")
    print("ALL OK")


if __name__ == "__main__":
    main()
