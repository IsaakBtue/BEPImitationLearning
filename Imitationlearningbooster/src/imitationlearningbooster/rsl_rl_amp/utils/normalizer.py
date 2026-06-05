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
        """Update running mean/variance with new values (numpy array)."""
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        values = values.reshape(-1, *self.shape)
        n = values.shape[0]
        new_mean = values.mean(axis=0)
        new_var = values.var(axis=0)
        self.count += n
        delta = new_mean - self.mean.cpu().numpy()
        self.mean = torch.tensor(
            self.mean.cpu().numpy() + delta * n / self.count, device=self.device
        )
        m_a = self.var.cpu().numpy() * n
        m_b = new_var * n
        M2 = m_a + m_b + delta**2 * n * self.count / (self.count + n)
        self.var = torch.tensor(M2 / (self.count + n), device=self.device)

    def normalize_torch(self, values, device):
        """Normalize torch tensor using running statistics."""
        mean = self.mean.to(device)
        var = self.var.to(device)
        return (values - mean) / torch.sqrt(var + 1e-8)

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
