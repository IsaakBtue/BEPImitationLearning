# src/simple_goalkeeper_him/rsl_rl_amp/modules/him_actor.py
"""HIM actor for SimpleGoalKeeperHim.

Faithful port of G1 ActorCritic (Humanoid-Goalkeeper rsl_rl/modules/actor_critic.py).

Architecture (G1 L118-154, L225-256):
  Three auxiliary networks all operate on the full obs history:
    history_encoder:  history → 16D latent
    ball_estimator:   history → 6D ball pos+vel estimate (MSE-supervised by privileged critic)
    region_estimator: history → num_regions-class logits (CE-supervised by GT region ID)

  Actor input (G1 L232):
    [cur_step_obs | history_latent(16) | ball_est(6) | argmax(region)(1)]

  Supervision losses (G1 him_ppo.py L186-229):
    est_loss    = MSE(ball_estimator(history), gt_ball)
    region_loss = CrossEntropy(region_estimator(history), gt_region_class)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import MLP, EmpiricalNormalization, GaussianDistribution


class HimActorModel(nn.Module):
    """HIM actor: history encoder + ball estimator + region estimator."""

    HISTORY_LATENT_DIM: int = 16
    ESTIMATE_BALL_DIM:  int = 6

    def __init__(
        self,
        obs_dim: int,
        num_one_step_obs: int,
        history_length: int,
        action_dim: int,
        num_regions: int = 2,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        term_dims: list[int] | None = None,
    ) -> None:
        """
        Args:
            obs_dim:          Total stacked obs dim  (= num_one_step_obs * history_length).
            num_one_step_obs: Single-step obs dim     (e.g. 86 for headless T1).
            history_length:   History frames          (10).
            action_dim:       Robot DOF               (21 for headless T1).
            num_regions:      AMP discriminator count (2: left/right).
            hidden_dims:      Actor MLP hidden dims.
            activation:       Activation function name.
            term_dims:        Per-term single-frame dims in mjlab obs group order.
                              Required to correctly extract cur_obs from term-major
                              history layout (see _extract_cur_obs docstring).
        """
        super().__init__()
        self.num_one_step_obs = num_one_step_obs
        self.history_length   = history_length
        self.num_regions      = num_regions
        self.obs_groups: list[str] = ["actor"]   # set by runner if different
        # Per-term single-frame dims for correct cur_obs extraction from term-major layout.
        # mjlab stacks each term's history independently: [t1_h0..hH, t2_h0..hH, ...].
        # G1 stacks time-major: [all_terms_h0, all_terms_h1, ..., all_terms_hH].
        # Without term_dims, full_history[:, -86:] extracts the wrong slice.
        self._term_dims: list[int] = list(term_dims) if term_dims else []

        full_history_dim = obs_dim               # = num_one_step_obs * history_length
        him_input_dim = (
            num_one_step_obs
            + self.HISTORY_LATENT_DIM
            + self.ESTIMATE_BALL_DIM
            + 1            # argmax(region)
        )

        # Obs normalizer (identical shape to what MLPModel would build)
        self.obs_normalizer = EmpiricalNormalization(full_history_dim)

        # Three auxiliary networks (G1 actor_critic.py L132-154)
        self.history_encoder = nn.Sequential(
            nn.Linear(full_history_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),              nn.ReLU(),
            nn.Linear(64, self.HISTORY_LATENT_DIM),
        )
        self.ball_estimator = nn.Sequential(
            nn.Linear(full_history_dim, 128), nn.ReLU(),
            nn.Linear(128, 32),              nn.ReLU(),
            nn.Linear(32, self.ESTIMATE_BALL_DIM),
        )
        self.region_estimator = nn.Sequential(
            nn.Linear(full_history_dim, 128), nn.ReLU(),
            nn.Linear(128, 32),              nn.ReLU(),
            nn.Linear(32, num_regions),
        )

        # Gaussian distribution (scalar std, learnable)
        self.distribution = GaussianDistribution(
            action_dim, init_std=1.0, std_type="scalar"
        )

        # Actor MLP: him_input_dim → action_dim (via distribution.input_dim)
        self.him_mlp = MLP(him_input_dim, self.distribution.input_dim, hidden_dims, activation)

        # Cached intermediate values read by runner for supervision losses
        self._last_ball_est:      torch.Tensor | None = None
        self._last_region_logits: torch.Tensor | None = None

    # ── Core forward ────────────────────────────────────────────────────────

    def _extract_cur_obs(self, full_history: torch.Tensor) -> torch.Tensor:
        """Extract most-recent single-frame obs from mjlab's term-major history layout.

        mjlab CircularBuffer stores history oldest→newest per term, then concatenates
        terms: [term1_h0..hH, term2_h0..hH, ...]. G1 uses time-major: [all_t0, all_t1, ...].
        So full_history[:, -num_one_step_obs:] is WRONG in mjlab — it extracts the last
        86 dims of the last few terms' histories, not the current observation.

        Correct extraction: for each term i with single-frame dim d_i and block of d_i*H
        elements, the newest frame is the last d_i elements of that block.
        """
        if not self._term_dims:
            # Fallback to G1 convention if term_dims not provided (time-major assumed).
            return full_history[:, -self.num_one_step_obs:]

        H = self.history_length
        cur_parts = []
        offset = 0
        for d in self._term_dims:
            block_end = offset + d * H
            cur_parts.append(full_history[:, block_end - d : block_end])
            offset = block_end
        return torch.cat(cur_parts, dim=-1)  # (N, num_one_step_obs)

    def _build_him_input(self, obs) -> torch.Tensor:
        """Concatenate obs groups, normalise, run auxiliary networks, build HIM input."""
        obs_list = [obs[g] for g in self.obs_groups]
        full_history = torch.cat(obs_list, dim=-1)   # (N, full_history_dim)

        # Update normaliser stats in training mode (explicit update, not inside forward)
        if self.training:
            self.obs_normalizer.update(full_history.detach())
        full_history = self.obs_normalizer(full_history)

        history_latent  = self.history_encoder(full_history)   # (N, 16)
        ball_est        = self.ball_estimator(full_history)    # (N, 6)
        region_logits   = self.region_estimator(full_history)  # (N, num_regions)

        # Cache for loss computation before the smooth-loss second actor pass overwrites them
        self._last_ball_est      = ball_est
        self._last_region_logits = region_logits

        cur_obs = self._extract_cur_obs(full_history)          # (N, num_one_step_obs)
        region_idx = torch.argmax(region_logits, dim=-1, keepdim=True).float()   # (N, 1)
        return torch.cat([cur_obs, history_latent, ball_est, region_idx], dim=-1)

    def forward(
        self,
        obs,
        masks=None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        actor_input = self._build_him_input(obs)
        mlp_out = self.him_mlp(actor_input)
        if stochastic_output:
            self.distribution.update(mlp_out)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_out)

    # ── MLPModel interface (required by GoalkeeperAmpRunner._update_with_amp) ──

    def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        mean, std = self.distribution.params
        # GaussianDistribution with scalar std returns std as (action_dim,) not (N, action_dim).
        # RolloutStorage allocates distribution_params[i] with shape (T, *p.shape), so
        # (action_dim,) gives (T, action_dim) and flatten(0,1) gives (T*action_dim,) —
        # completely wrong when indexed by (T*N,) batch indices, causing garbage values
        # in get_kl_divergence → NaN loss → NaN std_param → sample() crash.
        # Expand std to (N, action_dim) so storage gets (T, N, action_dim) and
        # flatten(0,1) gives the correct (T*N, action_dim).
        if std.dim() < mean.dim():
            std = std.expand_as(mean).contiguous()
        return (mean, std)

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    # ── Recurrent-model stubs (required by rsl_rl algorithm) ────────────────

    def reset(self, dones=None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None) -> None:
        pass

    def update_normalization(self, obs) -> None:
        """Explicit normaliser update (called by runner after rollout if desired)."""
        if not self.training:
            return
        obs_list = [obs[g] for g in self.obs_groups]
        full_history = torch.cat(obs_list, dim=-1)
        self.obs_normalizer.update(full_history.detach())

    # JIT/ONNX export not implemented for HIM actor
    def as_jit(self):
        raise NotImplementedError("HIM actor JIT export not supported.")

    def as_onnx(self, verbose: bool = False):
        raise NotImplementedError("HIM actor ONNX export not supported.")
