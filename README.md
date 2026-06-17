# latent_pipeline

Nbody→Mgas conditional flow matching (FM) + Mgas→(Ω_m, σ8) normalizing flow (NF),
multi-suite CAMELS 128³. 

## Structure

```
config/config.yaml   data + FM model/training + inference + NF + heldout
model.py             FM net (ClassicUNet) + latent GasEncoder (SE-ResNet3D)
data.py              suite pools, norm stats, FM datasets (cache + lazy)
module.py            FlowMatchingModel (FM Lightning: latent, EMA, xcorr/pk)
train.py             FM training (multi-suite, DDP)
infer.py             FM sampling → physical Mgas sample_*.npy

nf/
  encoder.py flow.py ResNet3D encoder + ConditionalFlow (NSF)
  module.py          NF dataset/datamodule + LitNFRegressor
  train.py infer.py  NF train / whole-pool eval (--data_mode real|synth)
  ood.py             OOD test: NF on held-out feedback model (Magneticum)
  predict.py         posterior prediction API

prep_cache.py             pre-normalise FM cache (Nbody overdensity + Mgas)
prep_magneticum_cache.py  held-out Magneticum 128³ cache
plot_nf_2param.py         NF in-distribution (val-split) truth-vs-pred
run.sh submit.sh          SLURM (train | infer | nf_train | nf_infer | nf_ood)
```
