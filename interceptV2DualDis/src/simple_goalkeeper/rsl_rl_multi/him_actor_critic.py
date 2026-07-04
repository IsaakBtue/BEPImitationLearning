"""HIM-style actor-critic: observation-history encoder + ball/region estimator
heads feeding the actor, ported from Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/
actor_critic.py, adapted for SGK's 2D (XY-only) ball convention and 4 regions
instead of G1's 3D/6 regions. See docs/superpowers/specs/2026-07-02-multi-
discriminator-amp-design.md section B.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name: str) -> nn.Module:
    return {
        "elu": nn.ELU(), "selu": nn.SELU(), "relu": nn.ReLU(),
        "crelu": nn.ReLU(), "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
    }[act_name]


class HimActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_one_step_obs: int,
        actor_history_length: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        history_latent_dim: int = 16,
        estimate_ball_dim: int = 4,
        num_regions: int = 4,
        fixed_std: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(f"HimActorCritic.__init__ got unexpected arguments, ignored: {list(kwargs.keys())}")
        super().__init__()
        act = get_activation(activation)

        self.num_one_step_obs = num_one_step_obs
        self.actor_history_length = actor_history_length
        self.history_latent_dim = history_latent_dim
        self.estimate_ball_dim = estimate_ball_dim
        self.num_regions = num_regions
        self.fixed_std = fixed_std

        mlp_input_dim_h = num_one_step_obs * actor_history_length
        self.num_actor_input = num_one_step_obs + history_latent_dim + estimate_ball_dim + 1

        self.history_encoder = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, history_latent_dim),
        )
        self.ball_estimator = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, estimate_ball_dim),
        )
        self.region_estimator = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, num_regions),
        )

        actor_layers = [nn.Linear(self.num_actor_input, actor_hidden_dims[0]), act]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers += [nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]), act]
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), act]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers += [nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]), act]
        self.critic = nn.Sequential(*critic_layers)

        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution: Normal | None = None
        self.estimate_ball: torch.Tensor | None = None
        self.estimate_region: torch.Tensor | None = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _build_actor_input(self, obs_current: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        history_latent = self.history_encoder(obs_history)
        self.estimate_ball = self.ball_estimator(obs_history)
        self.estimate_region = self.region_estimator(obs_history)
        region_arg = torch.argmax(self.estimate_region, dim=-1, keepdim=True).float()
        return torch.cat([obs_current, history_latent, self.estimate_ball, region_arg], dim=-1)

    def update_distribution(self, obs_current: torch.Tensor, obs_history: torch.Tensor) -> None:
        actor_input = self._build_actor_input(obs_current, obs_history)
        mean = self.actor(actor_input)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, obs_current: torch.Tensor, obs_history: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(obs_current, obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_current: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        actor_input = self._build_actor_input(obs_current, obs_history)
        return self.actor(actor_input)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)
