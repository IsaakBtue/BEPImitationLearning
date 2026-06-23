# src/imitationlearningbooster/rsl_rl_amp/utils/normalizer.py
"""Normalizer for AMP observations. Ported from rsl_rl."""
import torch
import numpy as np


class EmpiricalNormalization:
    def __init__(self, shape, device="cpu", eps=1e-8):
        self.device = device
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = eps
        self.shape = shape

    def update(self, values):
        """Update running mean/variance with new values (Chan's parallel algorithm)."""
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        values = values.reshape(-1, *self.shape)
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)

        old_count = self.count
        tot_count = old_count + batch_count

        delta = batch_mean - self.mean.cpu().numpy()
        new_mean = self.mean.cpu().numpy() + delta * batch_count / tot_count

        # m_a uses old_count (before increment), m_b uses batch_count
        m_a = self.var.cpu().numpy() * old_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * old_count * batch_count / tot_count

        self.mean = torch.tensor(new_mean, device=self.device)
        self.var = torch.tensor(M2 / tot_count, device=self.device)
        self.count = tot_count

    def normalize_torch(self, values, device):
        """Normalize torch tensor using running statistics."""
        mean = self.mean.to(device)
        var = self.var.to(device)
        # Clamp to ±10 — mirrors G1 Normalizer.normalize_torch clip_obs=10.0.
        # Prevents extreme outlier values (e.g. near-zero variance DOFs early in
        # training) from destabilizing the discriminator.
        return torch.clamp((values - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)

    def state_dict(self):
        return {
            "mean": self.mean.cpu().numpy(),
            "var": self.var.cpu().numpy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state):
        self.mean = torch.tensor(state["mean"], device=self.device)
        self.var = torch.tensor(state["var"], device=self.device)
        self.count = state["count"]
