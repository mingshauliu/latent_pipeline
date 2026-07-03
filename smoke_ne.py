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


def mkcfg(use_ne, in_ch, out_ch, use_vel=False, encoder_base=4, latent_head="tanh",
          target_fields=None, zero_init_film=True, zero_init_out=True):
    data = dict(use_velocity=use_vel, resolution=D, box_size=25,
                crop_size=None, clamp_val=10, n_cosmo=2)
    if target_fields is not None:
        data["target_fields"] = target_fields
    else:
        data["use_ne"] = use_ne
    return dict(
        data=data,
        model=dict(in_channels=in_ch, base_channels=8, out_channels=out_ch, cosmo_dim=2,
                   latent_dim=8, variational=False, encoder_base=encoder_base, encoder_dropout=0.0,
                   circular_padding=True, norm_type="pixel", latent_head=latent_head,
                   zero_init_film=zero_init_film, zero_init_out=zero_init_out),
        training=dict(lr=2e-4, weight_decay=1e-3, noise_std=0.1, time_sampling="logitnormal",
                      max_epochs=10, xcorr_every_n_epochs=0, xcorr_num_steps=4,
                      scheduler="cosine", warmup_epochs=0, ema=dict(enabled=False)),
    )


def batch(use_ne, use_vel=False, n_extra=None):
    # n_extra overrides use_ne: emit that many extra-target channels before cosmo.
    n_t = n_extra if n_extra is not None else (1 if use_ne else 0)
    nb, mg, co = torch.randn(2, 1, D, D, D), torch.randn(2, 1, D, D, D), torch.randn(2, 2)
    out = [nb, mg]
    for _ in range(n_t):
        out.append(torch.randn(2, 1, D, D, D))    # extra target (before cosmo)
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

    # 6. FULL multi-task [Mgas,ne,T] + velocity: target_fields=[ne,T], in=5/out=3, encoder 3-ch.
    cfg6 = mkcfg(None, 5, 3, use_vel=True, target_fields=["ne", "T"], latent_head="mlp")
    m6 = FlowMatchingModel(cfg6).to(dev)
    assert m6.n_extra == 2 and m6.out_channels == 3 and m6.gas_encoder.stem.weight.shape[1] == 3
    b = tuple(t.to(dev) for t in batch(None, use_vel=True, n_extra=2))   # (nb,mg,ne,T,co,vel)
    l, _, p, _ = m6._step(b, augment=True, sample_latent=True); l.backward()
    s = m6.sample(torch.randn(2, 1, D, D, D, device=dev), torch.randn(2, 2, device=dev),
                  torch.zeros(2, 8, device=dev), num_steps=3,
                  vel=torch.randn(2, 1, D, D, D, device=dev))
    assert p.shape[1] == 3 and s.shape[1] == 3, (p.shape, s.shape)
    print(f"[6] ne+T+vel OK pred{tuple(p.shape)} loss{float(l):.4f} sample{tuple(s.shape)} in=5/out=3")

    # 6b. warm-load real ep486 (1-ch) -> [Mgas,ne,T]+vel (in=5/out=3, encoder base40 fresh).
    if os.path.exists(CKPT):
        cfg = mkcfg(None, 5, 3, use_vel=True, target_fields=["ne", "T"], encoder_base=40, latent_head="mlp")
        cfg["model"]["base_channels"] = 128
        mw = FlowMatchingModel(cfg)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        warm_load_partial(mw, ck["state_dict"])
        oc = mw.net.out_conv.weight
        print(f"    out_conv mgas|max|={oc[0].abs().max():.3e} ne|max|={oc[1].abs().max():.3e} "
              f"T|max|={oc[2].abs().max():.3e} (ne/T rows ~0)")
        mw = mw.to(dev)
        b = tuple(t.to(dev) for t in batch(None, use_vel=True, n_extra=2))
        l, _, p, _ = mw._step(b, augment=True, sample_latent=True); l.backward()
        print(f"[6b] warm-load 2->5 in / 1->3 out / encoder base16->40 fresh OK loss{float(l):.4f}")

    # 7. LIVE FiLM vs zero-init: latent MUST influence the output at init when live.
    #    This is the fix for latent collapse — zero-init FiLM feeds latent through a
    #    zeroed proj so it has no effect / no gradient at step 0.
    for zi, tag in [(True, "zero-init"), (False, "LIVE")]:
        m = FlowMatchingModel(mkcfg(None, 5, 3, use_vel=True, target_fields=["ne", "T"],
                                    encoder_base=8, latent_head="mlp",
                                    zero_init_film=zi, zero_init_out=zi)).to(dev).eval()
        x = torch.randn(2, 5, D, D, D, device=dev); t = torch.rand(2, device=dev)
        c = torch.randn(2, 2, device=dev)
        with torch.no_grad():
            o0 = m.net(x, t, c, torch.zeros(2, 8, device=dev))
            o1 = m.net(x, t, c, torch.randn(2, 8, device=dev))
        eff = float((o0 - o1).abs().mean()); omean = float(o0.abs().mean())
        print(f"[7] {tag:9s} out|mean|={omean:.3e} latent_effect={eff:.3e}")
        if zi:
            assert eff < 1e-6, f"zero-init should kill latent effect, got {eff}"
        else:
            assert eff > 1e-4, f"LIVE FiLM must let latent affect output, got {eff}"
    print("[7] LIVE FiLM lets latent affect output at init; zero-init kills it (as designed)")

    # 8. ema.include_encoder=False -> encoder LIVE: shadow/bake carry net.* only,
    #    default (True) still shadows gas_encoder.* (backward compat), ckpt reloads.
    for inc in (True, False):
        cfg = mkcfg(False, 2, 1)
        cfg["training"]["ema"] = dict(enabled=True, decay=0.9, warmup_steps=0,
                                      include_encoder=inc)
        m8 = FlowMatchingModel(cfg).to(dev)
        m8._ema_update(); m8._ema_update()
        keys = set(m8._ema_shadow)
        has_enc = any(k.startswith("gas_encoder.") for k in keys)
        assert any(k.startswith("net.") for k in keys)
        assert has_enc == inc, (inc, sorted(keys)[:3])
        # swap-in must leave encoder params untouched when excluded
        enc_w = m8.gas_encoder.stem.weight.detach().clone()
        m8._ema_swap_in()
        if not inc:
            assert torch.equal(enc_w, m8.gas_encoder.stem.weight.detach())
        m8._ema_swap_out()
        ck = {"state_dict": m8.state_dict()}
        m8.on_save_checkpoint(ck)
        assert any(k.startswith("gas_encoder.") for k in ck["ema_shadow"]) == inc
        if not inc:  # baked state_dict keeps the LIVE encoder bit-exact
            assert torch.equal(ck["state_dict"]["gas_encoder.stem.weight"].cpu(), enc_w.cpu())
        m8b = FlowMatchingModel(cfg)
        m8b.load_state_dict(ck["state_dict"])
        print(f"[8] ema include_encoder={inc} OK (shadow {'has' if has_enc else 'skips'} encoder, ckpt reloads)")
    print("ALL OK")


if __name__ == "__main__":
    main()
