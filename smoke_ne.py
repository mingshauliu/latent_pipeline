"""CPU/GPU smoke for the ne multi-task OUTPUT channel + backward compatibility.

Checks:
  1. baseline (use_ne=false): 3-tuple batch -> out_channels=1 flow, sample (B,1,...).
  2. ne (use_ne=true): 4-tuple batch -> out_channels=2 flow, grad reaches the 2-ch
     encoder stem, sample returns (B,2,...).
  3. warm-load (train.warm_load_partial) from the real dim8 best ep486 (1-ch) into a
     2-ch ne model: out_conv 1->2 (ne row zero), enc1 input 2->3 (role-remap), encoder
     stem 1->2. No crash, then a forward/backward runs.
  4. backward compat: an ne (out=2) ckpt's net/encoder shapes differ from baseline ->
     just confirm a baseline model still builds + samples (the use_ne flag gates it).

Run: python smoke_ne.py            (CPU ok; GPU via gpu_smoke_ne.sbatch)
"""
import torch
from module import FlowMatchingModel
from train import warm_load_partial

D = 16
CKPT = "latent-pipeline/ffdsq458/checkpoints/best-epoch=486-val_loss=0.012231.ckpt"


def mkcfg(use_ne, in_ch, out_ch, use_vel=False, encoder_base=4, latent_head="tanh"):
    return dict(
        data=dict(use_ne=use_ne, use_velocity=use_vel, resolution=D, box_size=25,
                  crop_size=None, clamp_val=10, n_cosmo=2),
        model=dict(in_channels=in_ch, base_channels=8, out_channels=out_ch, cosmo_dim=2,
                   latent_dim=8, variational=False, encoder_base=encoder_base, encoder_dropout=0.0,
                   circular_padding=True, norm_type="pixel", latent_head=latent_head),
        training=dict(lr=2e-4, weight_decay=1e-3, noise_std=0.1, time_sampling="logitnormal",
                      max_epochs=10, xcorr_every_n_epochs=0, xcorr_num_steps=4,
                      scheduler="cosine", warmup_epochs=0, ema=dict(enabled=False)),
    )


def batch(use_ne, use_vel=False):
    nb, mg, co = torch.randn(2, 1, D, D, D), torch.randn(2, 1, D, D, D), torch.randn(2, 2)
    out = [nb, mg]
    if use_ne:
        out.append(torch.randn(2, 1, D, D, D))   # ne (before cosmo)
    out.append(co)
    if use_vel:
        out.append(torch.randn(2, 1, D, D, D))    # vel (after cosmo)
    return tuple(out)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    # 1. baseline
    mb = FlowMatchingModel(mkcfg(False, 2, 1)).to(dev)
    b = tuple(x.to(dev) for x in batch(False))
    l, _, p, _ = mb._step(b, augment=True, sample_latent=True); l.backward()
    s = mb.sample(torch.randn(2, 1, D, D, D, device=dev), torch.randn(2, 2, device=dev),
                  torch.zeros(2, 8, device=dev), num_steps=3)
    assert p.shape[1] == 1 and s.shape[1] == 1, (p.shape, s.shape)
    print(f"[1] baseline OK pred{tuple(p.shape)} loss{float(l):.4f} sample{tuple(s.shape)}")

    # 2. ne multi-task
    mn = FlowMatchingModel(mkcfg(True, 3, 2)).to(dev)
    b = tuple(x.to(dev) for x in batch(True))
    l, _, p, _ = mn._step(b, augment=True, sample_latent=True); l.backward()
    g = mn.gas_encoder.stem.weight.grad
    assert mn.gas_encoder.stem.weight.shape[1] == 2
    assert g is not None and torch.isfinite(g).all()
    s = mn.sample(torch.randn(2, 1, D, D, D, device=dev), torch.randn(2, 2, device=dev),
                  torch.zeros(2, 8, device=dev), num_steps=3)
    assert p.shape[1] == 2 and s.shape[1] == 2, (p.shape, s.shape)
    print(f"[2] ne OK pred{tuple(p.shape)} loss{float(l):.4f} sample{tuple(s.shape)} enc_in=2 grad_ok")

    # 2b. COMBINED ne + velocity: input [x_t(2), nbody, vel] = in_ch 4, out 2.
    mc = FlowMatchingModel(mkcfg(True, 4, 2, use_vel=True)).to(dev)
    b = tuple(x.to(dev) for x in batch(True, use_vel=True))
    l, _, p, _ = mc._step(b, augment=True, sample_latent=True); l.backward()
    s = mc.sample(torch.randn(2, 1, D, D, D, device=dev), torch.randn(2, 2, device=dev),
                  torch.zeros(2, 8, device=dev), num_steps=3,
                  vel=torch.randn(2, 1, D, D, D, device=dev))
    assert p.shape[1] == 2 and s.shape[1] == 2, (p.shape, s.shape)
    print(f"[2b] ne+vel OK pred{tuple(p.shape)} loss{float(l):.4f} sample{tuple(s.shape)} in_ch=4")

    # 3. warm-load real ep486 (1-ch) -> 2-ch ne model (production base_channels=128).
    import os
    if os.path.exists(CKPT):
        # the REAL combined config: in=4 (x_t2+nbody+vel), out=2, encoder base 16->40 (fresh),
        # latent_head=mlp (encoder3D no-tanh head -> fresh Sequential proj, warm-load skips it).
        cfg = mkcfg(True, 4, 2, use_vel=True, encoder_base=40, latent_head="mlp")
        cfg["model"]["base_channels"] = 128
        mw = FlowMatchingModel(cfg)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        warm_load_partial(mw, ck["state_dict"])
        # out_conv ne row (index 1) must be zero-init (designed); Mgas row copied non-zero.
        oc = mw.net.out_conv.weight
        print(f"    out_conv mgas-row|max|={oc[0].abs().max():.3e} ne-row|max|={oc[1].abs().max():.3e} (ne row ~0)")
        # enc1.conv1 role-remap: nbody col (src idx1) -> dst idx2 (=out_channels); cols
        # 1 (ne x_t) and 3 (vel) fresh. Just confirm fwd/bwd runs.
        mw = mw.to(dev)
        b = tuple(x.to(dev) for x in batch(True, use_vel=True))
        l, _, p, _ = mw._step(b, augment=True, sample_latent=True); l.backward()
        print(f"[3] warm-load 2->4 in / 1->2 out / encoder base16->40 fresh OK, fwd/bwd loss{float(l):.4f}")
    else:
        print(f"[3] SKIP warm-load (ckpt not found: {CKPT})")

    # 4. backward compat: baseline still builds + samples independently.
    mb2 = FlowMatchingModel(mkcfg(False, 2, 1)).to(dev)
    s = mb2.sample(torch.randn(2, 1, D, D, D, device=dev), torch.randn(2, 2, device=dev),
                   torch.zeros(2, 8, device=dev), num_steps=3)
    assert s.shape[1] == 1
    print("[4] backward-compat baseline OK")

    # 5. latent_head variants: tanh bounded [-1,1]; raw/mlp unbounded; grad reaches proj.
    for h in ["tanh", "raw", "mlp"]:
        m = FlowMatchingModel(mkcfg(True, 3, 2, latent_head=h)).to(dev)
        assert m.gas_encoder.latent_head == h
        x = torch.randn(2, 2, D, D, D, device=dev)   # (Mgas, ne) encode input
        z = m.gas_encoder(x)
        assert z.shape == (2, 8), z.shape
        if h == "tanh":
            assert z.abs().max() <= 1.0 + 1e-5, z.abs().max()
        b = tuple(t.to(dev) for t in batch(True))
        l, _, p, _ = m._step(b, augment=True, sample_latent=True); l.backward()
        seq = type(m.gas_encoder.proj).__name__
        print(f"[5:{h}] OK z{tuple(z.shape)} |z|max {z.abs().max():.3f} proj={seq} loss{float(l):.4f}")
    print("ALL OK")


if __name__ == "__main__":
    main()
