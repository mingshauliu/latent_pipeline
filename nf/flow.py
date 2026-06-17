"""Conditional Normalizing Flow head for p(params | summary).

Vendored from ../upscaling/nf/flow.py. Wraps zuko NSF/MAF; standardizes targets
internally so per-dim scales differ.
"""

import torch
import torch.nn as nn
import zuko


class ConditionalFlow(nn.Module):
    def __init__(
        self,
        num_params=2,
        context_dim=128,
        hidden_dim=128,
        num_transforms=4,
        flow_type="nsf",
        bins=8,
        target_mean=None,
        target_std=None,
    ):
        super().__init__()
        self.num_params = num_params
        self.context_dim = context_dim

        if flow_type == "nsf":
            self.flow = zuko.flows.NSF(
                features=num_params,
                context=context_dim,
                transforms=num_transforms,
                bins=bins,
                hidden_features=[hidden_dim, hidden_dim],
                randperm=True,
            )
        elif flow_type == "maf":
            self.flow = zuko.flows.MAF(
                features=num_params,
                context=context_dim,
                transforms=num_transforms,
                hidden_features=[hidden_dim, hidden_dim, hidden_dim],
                randperm=True,
            )
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

        if target_mean is not None:
            self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
            self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))
        else:
            self.register_buffer("target_mean", torch.zeros(num_params))
            self.register_buffer("target_std", torch.ones(num_params))

    def normalize_targets(self, y):
        return (y - self.target_mean) / self.target_std

    def denormalize_targets(self, y_norm):
        return y_norm * self.target_std + self.target_mean

    def log_prob(self, context, y):
        y_norm = self.normalize_targets(y)
        dist = self.flow(context)
        lp = dist.log_prob(y_norm)
        return lp - self.target_std.log().sum()

    def sample(self, context, num_samples=1000):
        dist = self.flow(context)
        samples_norm = dist.sample((num_samples,))
        return self.denormalize_targets(samples_norm)

    def get_posterior_stats(self, context, num_samples=2000):
        samples = self.sample(context, num_samples)
        return samples.mean(dim=0), samples.std(dim=0)
