# src/imitationlearningbooster/rsl_rl_amp/modules/discriminator.py
"""AMP Discriminator — ported from ccrpRepo/AMP_mjlab rsl_rl/modules/discriminator.py.
Adapted: default input_dim=46 (T1 23-DOF × 2 consecutive frames)."""
import torch
import torch.nn as nn
from torch import autograd
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 46, amp_reward_coef: float = 0.1,
                 hidden_layer_sizes: tuple = (512, 256),
                 device: str = "cpu"):
        super().__init__()
        self.amp_reward_coef = amp_reward_coef  # raw AMP reward scale (0.1)
        # Blending (40% AMP + 60% task) happens in the runner, not here

        # Fix 2.2: wrap linear layers with spectral normalization (matches G1 upstream)
        amp_layers = []
        in_dim = input_dim
        for h in hidden_layer_sizes:
            amp_layers += [spectral_norm(nn.Linear(in_dim, h)), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*amp_layers)
        self.amp_linear = spectral_norm(nn.Linear(in_dim, 1))
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.amp_linear(self.trunk(x))

    def compute_grad_pen(self, expert_obs: torch.Tensor, lambda_: float = 5.0) -> torch.Tensor:
        """Gradient penalty on expert data only (one-sided). λ=5 effective (matches G1 upstream).
        Fix 2.1: removed erroneous * 0.1 multiplier that made effective lambda 0.5 instead of 5."""
        expert_obs = expert_obs.detach().requires_grad_(True)
        disc = self.amp_linear(self.trunk(expert_obs))
        ones = torch.ones_like(disc)
        grad = autograd.grad(
            outputs=disc, inputs=expert_obs, grad_outputs=ones,
            create_graph=True, retain_graph=True
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_amp_reward(self, state: torch.Tensor, normalizer=None) -> torch.Tensor:
        """Return raw AMP reward (before blending with task reward).
        Blending (40% AMP + 60% task) is done by the runner."""
        with torch.no_grad():
            self.eval()
            obs = normalizer.normalize_torch(state, state.device) if normalizer else state
            d = self(obs)
            amp_r = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * (d - 1.0).pow(2), min=0.0)
            self.train()
        return amp_r
